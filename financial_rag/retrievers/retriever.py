"""
Hybrid 混合检索器 — 调度层

职责: 编排 BM25 + Vector + RRF + Rerank + Filter 的检索流程。
不包含具体实现，全部委托给子模块:
- BM25Engine: 关键词检索
- VectorEngine: 语义检索
- rrf_fusion: 多通道融合
- apply_filters: 元数据过滤
- save/load_index: 持久化
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from financial_rag.retrievers.bm25_engine import BM25Engine
from financial_rag.retrievers.vector_engine import VectorEngine
from financial_rag.retrievers.fusion import rrf_fusion
from financial_rag.retrievers.filters import apply_filters
from financial_rag.retrievers.embedding_cache import EmbeddingCache
from financial_rag.retrievers import persistence
from financial_rag.core.ingestion_scorer import IngestionScoreCard

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    BM25 + Embedding + Rerank 混合检索引擎 — 调度层

    可选三种模式:
    1. 纯本地 (无 API): BM25 + Jaccard → RRF 融合
    2. 带 Embedding: BM25 + 真实向量检索 → RRF 融合
    3. 全链路 (推荐): BM25 + Embedding → Rerank 精排

    使用方式:
    >>> retriever = HybridRetriever()
    >>> retriever.index(documents)              # 建索引
    >>> results = retriever.search("营收增长")   # 检索
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        embedder: Any = None,
        reranker: Any = None,
        tokenizer: Any = None,
        chunker: Any = None,
        parser: Any = None,
        chroma_persist_dir: Optional[str] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        cfg = config or {}
        self.rrf_k = cfg.get("rrf_k", 60)
        self.bm25_weight = cfg.get("bm25_weight", 0.3)
        self.vector_weight = cfg.get("vector_weight", 0.7)

        self.embedder = embedder
        self.reranker = reranker
        self._chunker = chunker
        self._parser = parser
        self._emb_cache = embedding_cache or EmbeddingCache.get_instance()

        # 子引擎
        self._bm25 = BM25Engine(tokenizer=tokenizer)
        self._vector = VectorEngine(
            embedder=embedder, tokenizer=tokenizer,
            chroma_persist_dir=chroma_persist_dir,
            embedding_cache=self._emb_cache,
        )

        # 内部状态
        self.documents: List[Dict] = []
        self.doc_embeddings: Optional[List[List[float]]] = None
        self._last_ingestion_score: Optional[IngestionScoreCard] = None

    # ===================== 索引 =====================

    def index(self, documents: List[Dict],
              precompute_embeddings: bool = True,
              use_chunker: bool = True):
        """索引文档"""
        if use_chunker and self._chunker:
            documents = self._chunker.split_documents(documents)

        self.documents = documents
        self._bm25.build(documents)

        # 预计算 embedding + 索引到 Chroma（走缓存，仅新增文档调 API）
        if precompute_embeddings and self._vector.has_embedding:
            texts = [d.get("text", "") for d in documents]
            logger.info(f"预计算 {len(texts)} 个文档的 embedding...")
            all_embeddings = self._emb_cache.embed_texts(texts, self.embedder)
            self.doc_embeddings = all_embeddings
            # 索引到 Chroma (ANN 索引)
            self._vector.index(documents, all_embeddings)
            logger.info(f"Embedding 预计算完成，维度: {len(all_embeddings[0]) if all_embeddings else 0}")
        else:
            self.doc_embeddings = None

        logger.info(
            f"HybridRetriever: 已索引 {len(documents)} 篇文档"
            f"{' (含 embedding)' if self.doc_embeddings else ''}"
            f"{' (含 rerank)' if self.reranker else ''}"
        )

        # Ingestion scoring
        try:
            card = IngestionScoreCard()
            card.record_preprocessing(documents)
            empty = sum(1 for d in documents if len(d.get("text", "").strip()) < 10)
            total_vocab = len(set(t for toks in self._bm25._corpus_tokens for t in toks))
            card.record_tokenization(self._bm25._corpus_tokens)
            card.record_index(
                doc_count=len(documents),
                total_vocab=total_vocab,
                bm25_built=self._bm25.is_built,
                chroma_built=self._vector._collection is not None,
                embedding_dim=len(self.doc_embeddings[0]) if self.doc_embeddings else 0,
                empty_docs=empty,
            )
            card.compute()
            card.log_summary()
            self._last_ingestion_score = card
        except Exception as e:
            logger.warning(f"IngestionScoreCard failed: {e}")

    def get_ingestion_score(self) -> Optional[IngestionScoreCard]:
        """Return the last ingestion scorecard (after index() or add())"""
        return self._last_ingestion_score

    def add(self, documents: List[Dict], use_chunker: bool = True):
        """增量添加文档"""
        if use_chunker and self._chunker:
            documents = self._chunker.split_documents(documents)

        self.documents.extend(documents)
        self._bm25.build(self.documents)

        # 增量添加时也计算新文档的 embeddings + 索引到 Chroma（走缓存）
        if self.embedder and self._vector.has_embedding:
            texts = [d.get("text", "") for d in documents]
            new_embeddings = self._emb_cache.embed_texts(texts, self.embedder)
            # Chroma 增量索引
            self._vector.add(documents, new_embeddings)
            if self.doc_embeddings is not None:
                self.doc_embeddings.extend(new_embeddings)
            logger.info(f"增量 embedding + Chroma: +{len(new_embeddings)} 篇")

    # ===================== 检索 =====================

    def rebuild_incremental(self, documents: List[Dict], use_chunker: bool = True):
        """
        增量重建索引：对比新旧文档，只对变化部分调 API。

        与 clear() + index() 不同，本方法：
        1. 通过 doc_id 对比新旧文档集合
        2. 只对新增文档计算 embedding（走缓存，命中则 0 API 调用）
        3. Chroma 增量 add，不全量重建
        4. BM25 全量重建（纯本地，毫秒级）
        """
        if use_chunker and self._chunker:
            documents = self._chunker.split_documents(documents)

        # 通过 doc_id 对比新旧文档
        old_id_set = {
            d.get("meta", {}).get("doc_id")
            for d in self.documents if d.get("meta", {}).get("doc_id")
        }
        new_id_set = {
            d.get("meta", {}).get("doc_id")
            for d in documents if d.get("meta", {}).get("doc_id")
        }

        added_ids = new_id_set - old_id_set
        removed_ids = old_id_set - new_id_set

        # 无变化则跳过
        if not added_ids and not removed_ids:
            logger.info("rebuild_incremental: 无变化，跳过")
            return {"added": 0, "removed": 0, "unchanged": len(documents)}

        added_docs = [d for d in documents
                      if d.get("meta", {}).get("doc_id") in added_ids]
        removed_docs = [d for d in self.documents
                        if d.get("meta", {}).get("doc_id") in removed_ids]

        logger.info(
            f"rebuild_incremental: +{len(added_docs)} new, "
            f"-{len(removed_docs)} removed, "
            f"{len(documents) - len(added_docs)} unchanged"
        )

        # 1. 删除已移除的文档 (Chroma + 内存)
        if removed_docs and self._vector._collection is not None:
            self._vector.remove(removed_docs)

        # 2. 替换文档列表，重建 BM25
        self.documents = documents
        self._bm25.build(self.documents)

        # 3. 为新增文档计算 embedding（走缓存）+ 增量入 Chroma
        if added_docs and self.embedder and self._vector.has_embedding:
            texts = [d.get("text", "") for d in added_docs]
            new_embeddings = self._emb_cache.embed_texts(texts, self.embedder)
            self._vector.add(added_docs, new_embeddings)
            # 同步 doc_embeddings 列表（全量重建，因为顺序可能变了）
            if self.doc_embeddings is not None:
                all_texts = [d.get("text", "") for d in self.documents]
                self.doc_embeddings = self._emb_cache.embed_texts(all_texts, self.embedder)
            logger.info(
                f"rebuild_incremental: +{len(added_docs)} embedding (cache), "
                f"Chroma incremental add done"
            )

        return {
            "added": len(added_docs),
            "removed": len(removed_docs),
            "unchanged": len(documents) - len(added_docs),
        }


    def search(self, query: str, top_k: int = 10,
               use_rerank: bool = True, scorecard=None,
               filters: Optional[Dict] = None) -> List[Dict]:
        """混合检索入口"""
        t0 = time.time()
        soft_filter_keys = set()

        # 0. 查询解析
        parsed = None
        if self._parser:
            parsed = self._parser.parse(query)
            parsed_filters = parsed.get_filters()
            if parsed_filters:
                filters = {**parsed_filters, **(filters or {})}
                soft_filter_keys = set(parsed_filters.keys())
            logger.info(
                f"QueryParser: stock={parsed.stock_code or '-'} "
                f"date={parsed.date or parsed.date_range or '-'} "
                f"type={parsed.query_type} "
                f"keywords={[(t, w) for t, w in parsed.keywords[:5]]}"
                f"{' expand=' + str(parsed.expanded_terms[:6]) if parsed.expanded_terms else ''}"
            )

        # 1. 分词
        if parsed and parsed.keywords:
            query_tokens = parsed.get_weighted_terms()
        else:
            query_tokens = self._bm25.tokenize(query)

        if scorecard:
            avg_len = sum(len(t) for t in query_tokens) / max(len(query_tokens), 1)
            token_score = (min(1.0, len(query_tokens) / 10) * 0.4
                           + min(1.0, len(set(query_tokens)) / max(len(query_tokens), 1)) * 0.3
                           + min(1.0, avg_len / 3) * 0.3)
            scorecard.record_tokenization(
                score=token_score, token_count=len(query_tokens),
                unique_tokens=len(set(query_tokens)), avg_token_len=avg_len,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # 2+3. BM25 + Vector 并发检索
        vec_query = (parsed.expanded_query
                     if parsed and parsed.expanded_query else query)

        def _run_bm25():
            return self._bm25.search(self.documents, query, top_k * 2,
                                     query_tokens=query_tokens)

        def _run_vector():
            if self._vector.has_embedding:
                return self._vector.search_embedding(
                    self.documents, vec_query, top_k * 2,
                    doc_embeddings=self.doc_embeddings,
                    cache_callback=lambda embs: setattr(self, 'doc_embeddings', embs),
                )
            else:
                return self._vector.search_jaccard(
                    self.documents, vec_query, top_k * 2,
                    tokenize_fn=self._bm25.tokenize,
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            bm25_future = pool.submit(_run_bm25)
            vec_future = pool.submit(_run_vector)
            bm25_results = bm25_future.result()
            vector_results = vec_future.result()

        if scorecard and bm25_results:
            bm25_scores = [r["score"] for r in bm25_results]
            scorecard.record_bm25(
                result_count=len(bm25_results), top_score=bm25_scores[0],
                avg_score=sum(bm25_scores) / len(bm25_scores),
                query_terms=len(query_tokens),
                matched_terms=len(set(t for r in bm25_results
                                      for t in self._bm25.tokenize(r.get("text", ""))
                                      if t in query_tokens)),
                elapsed_ms=(time.time() - t0) * 1000,
            )
        elif scorecard:
            scorecard.record_bm25(0, 0.0, 0.0, len(query_tokens), 0,
                                  elapsed_ms=(time.time() - t0) * 1000)

        if scorecard:
            total_elapsed = (time.time() - t0) * 1000
            if vector_results:
                vec_scores = [r["score"] for r in vector_results]
                scorecard.record_vector(
                    result_count=len(vector_results),
                    top_similarity=vec_scores[0],
                    avg_similarity=sum(vec_scores) / len(vec_scores),
                    embedding_dim=getattr(self.embedder, 'dimensions', 0)
                    if self.embedder else 0,
                    elapsed_ms=total_elapsed,
                )
            else:
                scorecard.record_vector(0, 0.0, 0.0, elapsed_ms=total_elapsed)

        # 4. RRF 融合
        candidates, fusion_stats = rrf_fusion(
            [(bm25_results, self.bm25_weight, "bm25"),
             (vector_results, self.vector_weight, "vector")],
            top_k=top_k * 2,
            rrf_k=self.rrf_k,
        )

        if scorecard:
            scorecard.record_rrf(
                fused_count=len(candidates),
                bm25_count=len(bm25_results),
                vector_count=len(vector_results),
                consensus_count=fusion_stats.consensus_count,
                elapsed_ms=0,
            )

        # 5. Rerank 精排
        if use_rerank and self.reranker and candidates:
            candidates = self._rerank(query, candidates, top_k)
            if scorecard:
                rerank_scores = [c.get("score", 0) for c in candidates]
                high_count = sum(1 for c in candidates
                                 if c.get("relevance_level") == "high")
                scorecard.record_rerank(
                    result_count=len(candidates),
                    top_rerank_score=rerank_scores[0] if rerank_scores else 0.0,
                    avg_rerank_score=(sum(rerank_scores) / len(rerank_scores)
                                      if rerank_scores else 0.0),
                    high_count=high_count, elapsed_ms=0,
                )

        # 6. Metadata 过滤 (soft keys for auto-injected filters)
        if filters:
            before = len(candidates)
            candidates = apply_filters(candidates, filters,
                                       soft_keys=soft_filter_keys)
            logger.info(f"Metadata 过滤: {before} → {len(candidates)}")

        logger.info(
            f"Hybrid: BM25={len(bm25_results)}, "
            f"Vector={len(vector_results)}, "
            f"Fused={len(candidates)}, "
            f"Rerank={'ON' if (use_rerank and self.reranker) else 'OFF'}"
        )
        return candidates[:top_k]

    # ===================== Rerank =====================

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """qwen3-rerank 精排，失败则降级"""
        if not candidates:
            return []
        doc_texts = [c.get("text", "") for c in candidates]
        try:
            rerank_results = self.reranker.rerank(
                query=query, documents=doc_texts,
                top_n=min(top_k, len(candidates)),
            )
        except Exception as e:
            logger.warning(f"Rerank 不可用，降级: {e}")
            return candidates[:top_k]

        result = []
        for i, rr in enumerate(rerank_results):
            if rr.index < len(candidates):
                item = dict(candidates[rr.index])
                item["score"] = rr.score
                item["rerank_score"] = rr.score
                item["relevance_level"] = rr.relevance_level
                item["retriever"] = "hybrid_rerank"
                item["rank"] = i + 1
                result.append(item)
        return result

    # ===================== 辅助方法 =====================

    def search_with_scores(self, query: str, top_k: int = 10,
                           use_rerank: bool = True) -> Tuple[List[Dict], Any]:
        """带全链路打分的检索"""
        from financial_rag.core.scorer import PipelineScoreCard
        card = PipelineScoreCard(query=query)
        results = self.search(query, top_k=top_k, use_rerank=use_rerank, scorecard=card)
        return results, card

    def llm_rewrite_query(self, query: str, llm, parsed=None) -> str:
        """
        LLM 增强层: 对短查询做语义扩展

        仅在以下条件下触发:
        - query 较短 (< 15 字) 或规则扩展词 < 2 个
        - LLM 可用

        Returns: 改写后的查询字符串 (原 query + LLM 补充词)
        """
        # 条件检查
        rule_expand_count = len(parsed.expanded_terms) if parsed else 0
        if len(query) >= 15 and rule_expand_count >= 2:
            return query  # 查询已足够丰富，不需要 LLM 扩展

        try:
            from financial_rag.llm.caller import LLMCaller
            caller = LLMCaller(llm)
            resp = caller.call(
                f"查询: {query}\n请生成 2-3 个相关的搜索关键词，用空格分隔。只输出关键词，不要解释。",
                system="你是搜索查询扩展助手。根据用户查询生成相关搜索词。",
                max_tokens=50, temperature=0.0,
            )
            extra_terms = resp.content.strip()
            if extra_terms and len(extra_terms) < 100:
                rewritten = query + " " + extra_terms
                logger.info(f"LLM query rewrite: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.debug(f"LLM query rewrite skipped: {e}")
        return query

    def remove(self, indices: List[int]):
        """Remove documents by index positions.

        Filters out docs, removes from Chroma, rebuilds BM25.
        """
        if not indices:
            return
        remove_set = set(indices)
        # 从 Chroma 中删除
        removed_docs = [d for i, d in enumerate(self.documents) if i in remove_set]
        self._vector.remove(removed_docs)
        # 从内存中过滤
        self.documents = [d for i, d in enumerate(self.documents) if i not in remove_set]
        if self.doc_embeddings is not None:
            self.doc_embeddings = [
                emb for i, emb in enumerate(self.doc_embeddings) if i not in remove_set
            ]
        self._bm25.build(self.documents)
        logger.info(f"HybridRetriever: removed {len(remove_set)} docs, {len(self.documents)} remaining")

    def clear(self):
        """清空索引"""
        self.documents = []
        self.doc_embeddings = None
        self._bm25.clear()
        self._vector.clear()
        logger.info("HybridRetriever: 索引已清空")

    def save_index(self, path: str):
        """持久化索引 (Chroma 自动持久化向量，此处只保存文档 + 配置)"""
        persistence.save_index(
            path, self.documents, doc_embeddings=None,
            config={"rrf_k": self.rrf_k, "bm25_weight": self.bm25_weight,
                     "vector_weight": self.vector_weight},
        )

    def load_index(self, path: str):
        """加载索引 (Chroma 自动加载向量，此处只加载文档 + 配置)"""
        data = persistence.load_index(path)
        self.documents = data["documents"]
        cfg = data.get("config", {})
        self.rrf_k = cfg.get("rrf_k", self.rrf_k)
        self.bm25_weight = cfg.get("bm25_weight", self.bm25_weight)
        self.vector_weight = cfg.get("vector_weight", self.vector_weight)
        self._bm25.build(self.documents)

        # 如果 Chroma 已有持久化数据，会自动加载
        # 否则，如果有 embedder，重新计算 embedding 并索引到 Chroma（走缓存）
        if self._vector.has_embedding and not self._vector._chroma_indexed:
            texts = [d.get("text", "") for d in self.documents]
            if texts:
                logger.info(f"加载后重建 Chroma 索引: {len(texts)} 篇文档...")
                all_embeddings = self._emb_cache.embed_texts(texts, self.embedder)
                self.doc_embeddings = all_embeddings
                self._vector.index(self.documents, all_embeddings)
                logger.info(f"Chroma 索引重建完成，维度: {len(all_embeddings[0]) if all_embeddings else 0}")
        else:
            # Chroma 自动加载了向量，不需要 doc_embeddings
            self.doc_embeddings = data.get("doc_embeddings")


# ===================== jieba 分词器工厂 =====================

_has_jieba = False
try:
    import jieba
    _has_jieba = True
except ImportError:
    pass


def jieba_tokenizer() -> callable:
    """创建 jieba 分词函数（通过 DictionaryRegistry 注入全量领域词典）"""
    if not _has_jieba:
        raise ImportError("请安装 jieba: pip install jieba")

    from financial_rag.retrievers.dictionary_registry import get_registry
    reg = get_registry()
    reg.set_jieba(jieba)  # 注入全部 jieba_words（内置 + 外部 JSON）

    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        words = jieba.lcut(text.lower())
        return [w.strip() for w in words
                if w.strip() and not all(c in '，。！？、；：“”‘’（）…—·《》' for c in w)]

    return _tokenize
