"""
向量检索引擎 — Embedding 语义检索 + Jaccard 回退

职责:
- 真实 Embedding 检索（阿里 API）
- Jaccard 回退（无 API 时）
- 余弦相似度计算
"""
import math
from typing import Dict, List, Any, Optional


class VectorEngine:
    """向量检索引擎 — Embedding + Jaccard fallback"""

    def __init__(self, embedder: Any = None, tokenizer=None):
        """
        Args:
            embedder: DashScopeEmbedding 实例
            tokenizer: 分词函数（Jaccard 回退用）
        """
        self.embedder = embedder
        self._tokenizer = tokenizer
        self._has_embedding = embedder is not None

    @property
    def has_embedding(self) -> bool:
        return self._has_embedding

    def search_embedding(self, documents: List[Dict], query: str, top_k: int,
                         doc_embeddings: Optional[List[List[float]]] = None,
                         cache_callback=None) -> List[Dict]:
        """
        使用真实 Embedding 做语义检索

        Args:
            documents: 文档列表
            query: 查询文本
            top_k: 返回数量
            doc_embeddings: 预计算的文档向量
            cache_callback: 缓存回调 callback(embeddings) — 实时计算的向量存回去
        """
        if not documents:
            return []

        query_vec = self.embedder.embed_query(query)

        doc_vecs = doc_embeddings
        if not doc_vecs:
            texts = [d.get("text", "") for d in documents]
            doc_vecs = self.embedder.embed_documents(texts)
            # 缓存回去，避免重复调用 API
            if cache_callback:
                cache_callback(doc_vecs)

        scores = {}
        for i, dv in enumerate(doc_vecs):
            scores[i] = self._cosine(query_vec, dv)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**documents[doc_id], "score": score, "retriever": "vector", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

    def search_jaccard(self, documents: List[Dict], query: str, top_k: int,
                       tokenize_fn=None) -> List[Dict]:
        """无 API 时的 Jaccard 语义匹配"""
        if not documents:
            return []
        fn = tokenize_fn or self._tokenizer
        if fn is None:
            return []

        q_words = set(fn(query))
        if not q_words:
            return []

        scores = {}
        for doc_id, doc in enumerate(documents):
            dw = set(fn(doc.get("text", "")))
            if not dw:
                continue
            intersection = q_words & dw
            union = q_words | dw
            jaccard = len(intersection) / len(union) if union else 0
            length_penalty = min(1.0, 200 / max(len(dw), 1))
            scores[doc_id] = jaccard * 0.7 + length_penalty * 0.3

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**documents[doc_id], "score": score, "retriever": "vector_jaccard", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
