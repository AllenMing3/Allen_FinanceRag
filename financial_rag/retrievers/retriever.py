"""
Hybrid 混合检索器 — BM25 + Vector (阿里 Embedding) + Rerank 重排序

三个通道:
- BM25: 关键词精准匹配（本地倒排索引）
- Vector: 阿里 text-embedding-v3 语义检索
- Rerank: 阿里 gte-rerank 精排（替换 RRF 融合）

所有参数可配置，与金融业务完全脱钩
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict
import hashlib
import re
import math
import time
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    BM25 + Embedding + Rerank 混合检索引擎

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
        # ---- 阿里 API 注入 ----
        embedder: Any = None,            # DashScopeEmbedding 实例
        reranker: Any = None,            # DashScopeReranker 实例
        # ---- 分词器注入 ----
        tokenizer: Any = None,           # jieba / 自定义分词器
    ):
        cfg = config or {}
        # RRF 参数（无 rerank 时使用）
        self.rrf_k = cfg.get("rrf_k", 60)
        self.bm25_weight = cfg.get("bm25_weight", 0.3)
        self.vector_weight = cfg.get("vector_weight", 0.7)

        # 阿里 API 客户端
        self.embedder = embedder
        self.reranker = reranker

        # 分词器
        self._tokenizer = tokenizer

        # 内部状态
        self.documents: List[Dict] = []
        self.doc_embeddings: Optional[List[List[float]]] = None   # 预计算的文档向量
        self.bm25_index: Dict[str, List[int]] = defaultdict(list)
        self._has_embedding = embedder is not None
        self._has_rerank = reranker is not None

    # ===================== 索引 =====================

    def index(self, documents: List[Dict], precompute_embeddings: bool = True):
        """
        索引文档

        Args:
            documents: [{"text": "...", "meta": ...}, ...]
            precompute_embeddings: 是否预计算文档向量（有 embedder 时默认开启）
        """
        self.documents = documents
        self._build_bm25_index()

        # 预计算文档向量
        if precompute_embeddings and self._has_embedding:
            texts = [d.get("text", "") for d in documents]
            logger.info(f"预计算 {len(texts)} 个文档的 embedding...")
            # DashScope batch limit: 10 per request
            all_embeddings = []
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self.embedder.embed_documents(batch)
                all_embeddings.extend(resp)
            self.doc_embeddings = all_embeddings
            logger.info(f"Embedding 预计算完成，维度: {len(all_embeddings[0]) if all_embeddings else 0}")
        else:
            self.doc_embeddings = None

        logger.info(
            f"HybridRetriever: 已索引 {len(documents)} 篇文档"
            f"{' (含 embedding)' if self.doc_embeddings else ''}"
            f"{' (含 rerank)' if self._has_rerank else ''}"
        )

    def add(self, documents: List[Dict]):
        """增量添加文档"""
        start = len(self.documents)
        self.documents.extend(documents)
        for i, doc in enumerate(documents):
            for word in self._tokenize(doc.get("text", "")):
                self.bm25_index[word].append(start + i)

    # ===================== 检索 =====================

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
        scorecard = None,               # PipelineScoreCard 实例（可选）
    ) -> List[Dict]:
        """
        混合检索入口

        Args:
            query: 查询文本
            top_k: 返回数量
            use_rerank: 是否启用 Rerank 精排（需要已注入 reranker）
            scorecard: PipelineScoreCard 实例，传入则自动记录每个子阶段的分数

        Returns:
            排序后的文档列表
        """
        t0 = time.time()

        # ---- 0. 分词阶段 ----
        t_tok = time.time()
        query_tokens = self._tokenize(query)
        tok_elapsed = (time.time() - t_tok) * 1000
        if scorecard:
            # 分词评分: 基于分词数量 & 唯一性 & 平均长度
            avg_len = sum(len(t) for t in query_tokens) / max(len(query_tokens), 1)
            token_score = min(1.0, len(query_tokens) / 10) * 0.4 \
                        + min(1.0, len(set(query_tokens)) / max(len(query_tokens), 1)) * 0.3 \
                        + min(1.0, avg_len / 3) * 0.3
            scorecard.record_tokenization(
                score=token_score,
                token_count=len(query_tokens),
                unique_tokens=len(set(query_tokens)),
                avg_token_len=avg_len,
                elapsed_ms=tok_elapsed,
            )

        # 1. BM25 关键词检索
        bm25_results = self._bm25_search(query, top_k * 2, query_tokens=query_tokens)
        if scorecard and bm25_results:
            bm25_scores = [r["score"] for r in bm25_results]
            scorecard.record_bm25(
                result_count=len(bm25_results),
                top_score=bm25_scores[0],
                avg_score=sum(bm25_scores) / len(bm25_scores),
                query_terms=len(query_tokens),
                matched_terms=len(set(t for r in bm25_results for t in self._tokenize(r.get("text", "")) if t in query_tokens)),
                elapsed_ms=(time.time() - t0) * 1000 - tok_elapsed,
            )
        elif scorecard:
            scorecard.record_bm25(0, 0.0, 0.0, len(query_tokens), 0,
                                  elapsed_ms=(time.time() - t0) * 1000 - tok_elapsed)

        # 2. Vector 语义检索
        t_vec = time.time()
        if self._has_embedding:
            vector_results = self._vector_search_real(query, top_k * 2)
        else:
            vector_results = self._vector_search_simple(query, top_k * 2)
        vec_elapsed = (time.time() - t_vec) * 1000

        if scorecard:
            if vector_results:
                vec_scores = [r["score"] for r in vector_results]
                scorecard.record_vector(
                    result_count=len(vector_results),
                    top_similarity=vec_scores[0],
                    avg_similarity=sum(vec_scores) / len(vec_scores),
                    embedding_dim=getattr(self.embedder, 'dimensions', 0) if self._has_embedding else 0,
                    elapsed_ms=vec_elapsed,
                )
            else:
                scorecard.record_vector(0, 0.0, 0.0, elapsed_ms=vec_elapsed)

        # 3. RRF 融合 → 候选集
        t_rrf = time.time()
        candidates = self._rrf_fusion(bm25_results, vector_results, top_k * 2)
        rrf_elapsed = (time.time() - t_rrf) * 1000

        if scorecard:
            # 计算共识：在两个检索器中都出现的文档数
            bm25_texts = set(r.get("text", "") for r in bm25_results)
            vec_texts = set(r.get("text", "") for r in vector_results)
            consensus = len(bm25_texts & vec_texts)
            scorecard.record_rrf(
                fused_count=len(candidates),
                bm25_count=len(bm25_results),
                vector_count=len(vector_results),
                consensus_count=consensus,
                elapsed_ms=rrf_elapsed,
            )

        # 4. Rerank 精排（如果开启了）
        if use_rerank and self._has_rerank and candidates:
            t_rerank = time.time()
            candidates = self._rerank(query, candidates, top_k)
            rerank_elapsed = (time.time() - t_rerank) * 1000

            if scorecard:
                high_count = sum(1 for c in candidates if c.get("relevance_level") == "high")
                rerank_scores = [c.get("score", 0) for c in candidates]
                scorecard.record_rerank(
                    result_count=len(candidates),
                    top_rerank_score=rerank_scores[0] if rerank_scores else 0.0,
                    avg_rerank_score=sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0,
                    high_count=high_count,
                    elapsed_ms=rerank_elapsed,
                )

        logger.info(
            f"Hybrid: BM25={len(bm25_results)}, "
            f"Vector={len(vector_results)}, "
            f"Fused={len(candidates)}, "
            f"Rerank={'ON' if (use_rerank and self._has_rerank) else 'OFF'}"
        )

        return candidates[:top_k]

    # ===================== BM25 (本地，无需 API) =====================

    def _tokenize(self, text: str) -> List[str]:
        """分词 — 优先用注入的分词器 (jieba)，否则回退到正则"""
        if self._tokenizer is not None:
            try:
                return self._tokenizer(text)
            except Exception:
                pass
        # 回退: 中文按字符序列，英文按单词
        tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+(?:\.\d+)?[%％]?', text.lower())
        return tokens

    def _build_bm25_index(self):
        self.bm25_index.clear()
        for doc_id, doc in enumerate(self.documents):
            for word in set(self._tokenize(doc.get("text", ""))):
                self.bm25_index[word].append(doc_id)

    def _bm25_search(self, query: str, top_k: int, query_tokens: List[str] = None) -> List[Dict]:
        if not self.documents:
            return []
        query_terms = query_tokens if query_tokens is not None else self._tokenize(query)
        if not query_terms:
            return []

        N = len(self.documents)
        avg_dl = sum(len(d.get("text", "")) for d in self.documents) / max(N, 1)
        k1, b = 1.2, 0.75

        scores: Dict[int, float] = {}
        for term in query_terms:
            if term not in self.bm25_index:
                continue
            doc_ids = self.bm25_index[term]
            df = len(doc_ids)
            idf = math.log1p((N - df + 0.5) / (df + 0.5))

            for doc_id in doc_ids:
                text = self.documents[doc_id].get("text", "")
                dl = len(text)
                tf = text.lower().count(term)
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                scores[doc_id] = scores.get(doc_id, 0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**self.documents[doc_id], "score": score, "retriever": "bm25", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

    # ===================== Vector: 真实 Embedding (阿里 API) =====================

    def _vector_search_real(self, query: str, top_k: int) -> List[Dict]:
        """使用阿里 text-embedding-v3 做真实语义检索"""
        if not self.documents:
            return []

        # 获取 query embedding
        query_vec = self.embedder.embed_query(query)

        # 如果没有预计算的文档向量，实时计算
        doc_vecs = self.doc_embeddings
        if not doc_vecs:
            texts = [d.get("text", "") for d in self.documents]
            resp = self.embedder.embed_documents(texts)
            doc_vecs = resp

        # 余弦相似度
        scores = {}
        for i, dv in enumerate(doc_vecs):
            scores[i] = self._cosine(query_vec, dv)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**self.documents[doc_id], "score": score, "retriever": "vector", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

    # ===================== Vector: 简易 Jaccard (无 API fallback) =====================

    def _vector_search_simple(self, query: str, top_k: int) -> List[Dict]:
        """无 API 时的简易 Jaccard 语义匹配"""
        if not self.documents:
            return []
        q_words = set(self._tokenize(query))
        if not q_words:
            return []

        scores = {}
        for doc_id, doc in enumerate(self.documents):
            text = doc.get("text", "")
            dw = set(self._tokenize(text))
            if not dw:
                continue
            intersection = q_words & dw
            union = q_words | dw
            jaccard = len(intersection) / len(union) if union else 0
            length_penalty = min(1.0, 200 / max(len(dw), 1))
            scores[doc_id] = jaccard * 0.7 + length_penalty * 0.3

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**self.documents[doc_id], "score": score, "retriever": "vector_jaccard", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ===================== RRF 融合 =====================

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """RRF 融合排序 — 作为 Rerank 前的候选集"""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for results, weight in [(bm25_results, self.bm25_weight),
                                 (vector_results, self.vector_weight)]:
            for r in results:
                text = r.get("text", "")
                doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
                rank = r.get("rank", 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (self.rrf_k + rank)
                doc_map[doc_id] = r

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = []
        for rank_i, (doc_id, rrf_score) in enumerate(ranked):
            item = dict(doc_map[doc_id])
            item["rrf_score"] = rrf_score
            item["rank"] = rank_i + 1
            result.append(item)
        return result

    # ===================== Rerank: 阿里 gte-rerank =====================

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        使用阿里 gte-rerank 对候选文档精排

        这是三大架构中 Indexer 的最后一环：
        BM25 + Embedding → RRF → Rerank → 最终 Top-K

        如果 Rerank API 不可用（403/401 等），自动降级为 RRF 融合结果。
        """
        if not candidates:
            return []

        # 提取文档文本列表
        doc_texts = [c.get("text", "") for c in candidates]

        try:
            # 调用阿里 Rerank API
            rerank_results = self.reranker.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(candidates)),
            )
        except Exception as e:
            logger.warning(f"Rerank 不可用，降级为 RRF 融合结果: {e}")
            return candidates[:top_k]

        # 重新组装结果，注入 rerank 分数
        result = []
        for i, rr in enumerate(rerank_results):
            if rr.index < len(candidates):
                item = dict(candidates[rr.index])
                item["score"] = rr.score
                item["rerank_score"] = rr.score
                item["relevance_level"] = rr.relevance_level
                item["retriever"] = f"hybrid_rerank"
                item["rank"] = i + 1
                result.append(item)

        return result

    def search_with_scores(self, query: str, top_k: int = 10,
                           use_rerank: bool = True) -> Tuple[List[Dict], "PipelineScoreCard"]:
        """
        带全链路打分的检索 — 自动创建打分卡并返回

        Returns:
            (检索结果, PipelineScoreCard 打分卡)
        """
        from financial_rag.core.scorer import PipelineScoreCard
        card = PipelineScoreCard(query=query)
        results = self.search(query, top_k=top_k, use_rerank=use_rerank, scorecard=card)
        return results, card

    def clear(self):
        """清空索引"""
        self.documents = []
        self.doc_embeddings = None
        self.bm25_index.clear()
        logger.info("HybridRetriever: 索引已清空")


# ===================== jieba 分词器工厂 =====================

_has_jieba = False
try:
    import jieba
    _has_jieba = True
except ImportError:
    pass


def jieba_tokenizer() -> callable:
    """
    创建 jieba 分词函数

    第一次调用自动加载默认词典 + 金融词典
    返回: callable(text) -> List[str]
    """
    if not _has_jieba:
        raise ImportError("请安装 jieba: pip install jieba")

    jieba.setLogLevel(20)  # 抑制 jieba 日志

    # 添加常见金融术语到词典
    finance_words = [
        "营业收入", "净利润", "毛利率", "净资产收益率", "每股收益",
        "经营活动现金流", "总资产", "总负债", "资产负债率",
        "同比增长", "环比增长", "基本每股收益", "加权平均",
        "归母净利润", "扣非净利润", "流动资产", "非流动资产",
        "流动负债", "非流动负债", "所有者权益", "少数股东权益",
        "应收账款", "存货", "固定资产", "无形资产", "商誉",
        "短期借款", "长期借款", "应付票据", "应付账款",
        "销售费用", "管理费用", "财务费用", "研发费用",
        "投资收益", "公允价值变动", "信用减值损失",
        "经营活动", "投资活动", "筹资活动", "汇率变动",
        "贵州茅台", "五粮液", "宁德时代", "比亚迪",
        "央行", "降准", "降息", "LPR", "MLF", "逆回购",
        "上证指数", "深证成指", "创业板指", "科创板",
    ]
    for w in finance_words:
        jieba.add_word(w)

    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        words = jieba.lcut(text.lower())
        # 过滤纯标点和空字符串
        return [w.strip() for w in words if w.strip() and not all(c in '，。！？、；：""''（）…—·《》' for c in w)]

    return _tokenize
