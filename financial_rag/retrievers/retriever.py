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

from financial_rag.retrievers.bm25_engine import BM25Engine
from financial_rag.retrievers.vector_engine import VectorEngine
from financial_rag.retrievers.fusion import rrf_fusion
from financial_rag.retrievers.filters import apply_filters
from financial_rag.retrievers import persistence

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
    ):
        cfg = config or {}
        self.rrf_k = cfg.get("rrf_k", 60)
        self.bm25_weight = cfg.get("bm25_weight", 0.3)
        self.vector_weight = cfg.get("vector_weight", 0.7)

        self.embedder = embedder
        self.reranker = reranker
        self._chunker = chunker
        self._parser = parser

        # 子引擎
        self._bm25 = BM25Engine(tokenizer=tokenizer)
        self._vector = VectorEngine(embedder=embedder, tokenizer=tokenizer)

        # 内部状态
        self.documents: List[Dict] = []
        self.doc_embeddings: Optional[List[List[float]]] = None

    # ===================== 索引 =====================

    def index(self, documents: List[Dict],
              precompute_embeddings: bool = True,
              use_chunker: bool = True):
        """索引文档"""
        if use_chunker and self._chunker:
            documents = self._chunker.split_documents(documents)

        self.documents = documents
        self._bm25.build(documents)

        # 预计算 embedding
        if precompute_embeddings and self._vector.has_embedding:
            texts = [d.get("text", "") for d in documents]
            logger.info(f"预计算 {len(texts)} 个文档的 embedding...")
            all_embeddings = []
            for i in range(0, len(texts), 10):
                all_embeddings.extend(
                    self.embedder.embed_documents(texts[i:i + 10])
                )
            self.doc_embeddings = all_embeddings
            logger.info(f"Embedding 预计算完成，维度: {len(all_embeddings[0]) if all_embeddings else 0}")
        else:
            self.doc_embeddings = None

        logger.info(
            f"HybridRetriever: 已索引 {len(documents)} 篇文档"
            f"{' (含 embedding)' if self.doc_embeddings else ''}"
            f"{' (含 rerank)' if self.reranker else ''}"
        )

    def add(self, documents: List[Dict], use_chunker: bool = True):
        """增量添加文档"""
        if use_chunker and self._chunker:
            documents = self._chunker.split_documents(documents)

        self.documents.extend(documents)
        self._bm25.build(self.documents)

        # Bug fix: 增量添加时也计算新文档的 embeddings
        if self.embedder and self.doc_embeddings is not None:
            texts = [d.get("text", "") for d in documents]
            new_embeddings = []
            for i in range(0, len(texts), 10):
                new_embeddings.extend(
                    self.embedder.embed_documents(texts[i:i + 10])
                )
            self.doc_embeddings.extend(new_embeddings)
            logger.info(f"增量 embedding: +{len(new_embeddings)} 篇")

    # ===================== 检索 =====================

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

        # 2. BM25 检索
        bm25_results = self._bm25.search(self.documents, query, top_k * 2,
                                          query_tokens=query_tokens)
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

        # 3. Vector 检索
        t_vec = time.time()
        if self._vector.has_embedding:
            vector_results = self._vector.search_embedding(
                self.documents, query, top_k * 2,
                doc_embeddings=self.doc_embeddings,
                cache_callback=lambda embs: setattr(self, 'doc_embeddings', embs),
            )
        else:
            vector_results = self._vector.search_jaccard(
                self.documents, query, top_k * 2,
                tokenize_fn=self._bm25.tokenize,
            )
        vec_elapsed = (time.time() - t_vec) * 1000

        if scorecard:
            if vector_results:
                vec_scores = [r["score"] for r in vector_results]
                scorecard.record_vector(
                    result_count=len(vector_results),
                    top_similarity=vec_scores[0],
                    avg_similarity=sum(vec_scores) / len(vec_scores),
                    embedding_dim=getattr(self.embedder, 'dimensions', 0)
                    if self.embedder else 0,
                    elapsed_ms=vec_elapsed,
                )
            else:
                scorecard.record_vector(0, 0.0, 0.0, elapsed_ms=vec_elapsed)

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

    def clear(self):
        """清空索引"""
        self.documents = []
        self.doc_embeddings = None
        self._bm25.clear()
        logger.info("HybridRetriever: 索引已清空")

    def save_index(self, path: str):
        """持久化索引"""
        persistence.save_index(
            path, self.documents, self.doc_embeddings,
            config={"rrf_k": self.rrf_k, "bm25_weight": self.bm25_weight,
                     "vector_weight": self.vector_weight},
        )

    def load_index(self, path: str):
        """加载索引"""
        data = persistence.load_index(path)
        self.documents = data["documents"]
        self.doc_embeddings = data.get("doc_embeddings")
        cfg = data.get("config", {})
        self.rrf_k = cfg.get("rrf_k", self.rrf_k)
        self.bm25_weight = cfg.get("bm25_weight", self.bm25_weight)
        self.vector_weight = cfg.get("vector_weight", self.vector_weight)
        self._bm25.build(self.documents)


# ===================== jieba 分词器工厂 =====================

_has_jieba = False
try:
    import jieba
    _has_jieba = True
except ImportError:
    pass


def jieba_tokenizer() -> callable:
    """创建 jieba 分词函数（自动加载金融词典）"""
    if not _has_jieba:
        raise ImportError("请安装 jieba: pip install jieba")

    from financial_rag.retrievers.dictionaries import JIEBA_FINANCE_WORDS

    jieba.setLogLevel(20)
    for w in JIEBA_FINANCE_WORDS:
        jieba.add_word(w)

    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        words = jieba.lcut(text.lower())
        return [w.strip() for w in words
                if w.strip() and not all(c in '，。！？、；：""''（）…—·《》' for c in w)]

    return _tokenize
