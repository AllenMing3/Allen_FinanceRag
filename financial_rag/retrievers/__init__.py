"""
Hybrid 混合检索器 — BM25 + Vector (阿里 Embedding) + Rerank 重排序

三个通道:
- BM25: 关键词精准匹配（本地倒排索引）
- Vector: 阿里 text-embedding-v3 语义检索
- Rerank: 阿里 gte-rerank 精排（替换 RRF 融合）

所有参数可配置，与金融业务完全脱钩
"""
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict
import hashlib
import re
import math
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
    ):
        cfg = config or {}
        # RRF 参数（无 rerank 时使用）
        self.rrf_k = cfg.get("rrf_k", 60)
        self.bm25_weight = cfg.get("bm25_weight", 0.3)
        self.vector_weight = cfg.get("vector_weight", 0.7)

        # 阿里 API 客户端
        self.embedder = embedder
        self.reranker = reranker

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
            resp = self.embedder.embed_documents(texts)
            self.doc_embeddings = resp
            logger.info(f"Embedding 预计算完成，维度: {len(resp[0]) if resp else 0}")
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
    ) -> List[Dict]:
        """
        混合检索入口

        Args:
            query: 查询文本
            top_k: 返回数量
            use_rerank: 是否启用 Rerank 精排（需要已注入 reranker）

        Returns:
            排序后的文档列表
        """
        # 1. BM25 关键词检索
        bm25_results = self._bm25_search(query, top_k * 2)

        # 2. Vector 语义检索
        if self._has_embedding:
            vector_results = self._vector_search_real(query, top_k * 2)
        else:
            vector_results = self._vector_search_simple(query, top_k * 2)

        # 3. RRF 融合 → 候选集
        candidates = self._rrf_fusion(bm25_results, vector_results, top_k * 2)

        # 4. Rerank 精排（如果开启了）
        if use_rerank and self._has_rerank and candidates:
            candidates = self._rerank(query, candidates, top_k)

        logger.info(
            f"Hybrid: BM25={len(bm25_results)}, "
            f"Vector={len(vector_results)}, "
            f"Fused={len(candidates)}, "
            f"Rerank={'ON' if (use_rerank and self._has_rerank) else 'OFF'}"
        )

        return candidates[:top_k]

    # ===================== BM25 (本地，无需 API) =====================

    def _tokenize(self, text: str) -> List[str]:
        """分词 — 中文按字符序列，英文按单词"""
        tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+(?:\.\d+)?[%％]?', text.lower())
        return tokens

    def _build_bm25_index(self):
        self.bm25_index.clear()
        for doc_id, doc in enumerate(self.documents):
            for word in set(self._tokenize(doc.get("text", ""))):
                self.bm25_index[word].append(doc_id)

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        if not self.documents:
            return []
        query_terms = self._tokenize(query)
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
        """
        if not candidates:
            return []

        # 提取文档文本列表
        doc_texts = [c.get("text", "") for c in candidates]

        # 调用阿里 Rerank API
        rerank_results = self.reranker.rerank(
            query=query,
            documents=doc_texts,
            top_n=min(top_k, len(candidates)),
        )

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

    def clear(self):
        """清空索引"""
        self.documents = []
        self.doc_embeddings = None
        self.bm25_index.clear()
        logger.info("HybridRetriever: 索引已清空")
