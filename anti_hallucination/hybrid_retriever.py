"""
Hybrid RAG 检索器: BM25 + Vector + RRF 融合

特点:
1. BM25 关键词检索 - 精准匹配
2. Vector 语义检索 - 语义理解
3. RRF 融合排序 - 综合排序
"""
from typing import Dict, Any, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid RAG 检索器

    检索流程:
    1. BM25 关键词检索
    2. Vector 语义检索
    3. RRF (Reciprocal Rank Fusion) 融合排序
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.rrf_k = self.config.get("rrf_k", 60)
        self.bm25_weight = self.config.get("bm25_weight", 0.3)
        self.vector_weight = self.config.get("vector_weight", 0.7)

        # 知识库（实际项目中使用 ChromaDB/FAISS 等）
        self.documents: List[Dict] = []
        self.bm25_index: Dict[str, List[int]] = defaultdict(list)

        logger.info("HybridRetriever 初始化完成")

    def index_documents(self, documents: List[Dict]):
        """
        索引文档（BM25 + Vector）

        Args:
            documents: [{"text": "...", "metadata": {...}}, ...]
        """
        self.documents = documents
        self._build_bm25_index(documents)
        logger.info(f"已索引 {len(documents)} 篇文档")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Hybrid 检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            排序后的检索结果
        """
        # 1. BM25 检索
        bm25_results = self._bm25_search(query, top_k * 2)

        # 2. Vector 检索（简化：基于文本相似度）
        vector_results = self._vector_search(query, top_k * 2)

        # 3. RRF 融合
        fused = self._rrf_fusion(bm25_results, vector_results, top_k)

        logger.info(f"Hybrid检索: BM25={len(bm25_results)}, Vector={len(vector_results)}, Fused={len(fused)}")

        return fused

    def _build_bm25_index(self, documents: List[Dict]):
        """构建 BM25 倒排索引"""
        self.bm25_index.clear()

        for doc_id, doc in enumerate(documents):
            text = doc.get("text", "")
            words = set(self._tokenize(text))
            for word in words:
                self.bm25_index[word].append(doc_id)

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        import re
        return re.findall(r'\w+', text.lower())

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 关键词检索"""
        if not self.documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # 计算 BM25 分数（简化版）
        scores = {}
        N = len(self.documents)
        avg_dl = sum(len(d.get("text", "").split()) for d in self.documents) / max(N, 1)
        k1, b = 1.2, 0.75

        for term in query_terms:
            if term not in self.bm25_index:
                continue

            doc_ids = self.bm25_index[term]
            df = len(doc_ids)  # 文档频率
            idf = max(0, ((N - df + 0.5) / (df + 0.5)) + 1)

            for doc_id in doc_ids:
                text = self.documents[doc_id].get("text", "")
                dl = len(text.split())
                tf = text.lower().count(term)
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0) + score

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                **self.documents[doc_id],
                "score": score,
                "retriever": "bm25",
                "rank": i + 1,
            }
            for i, (doc_id, score) in enumerate(ranked)
        ]

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Vector 语义检索（简化版：基于关键词重叠的语义相似度）

        实际项目中替换为:
        - embedding 模型 + FAISS/ChromaDB 向量检索
        - 或使用 LlamaIndex 的 VectorStoreIndex
        """
        if not self.documents:
            return []

        query_words = set(self._tokenize(query))
        if not query_words:
            return []

        scores = {}
        for doc_id, doc in enumerate(self.documents):
            text = doc.get("text", "")
            doc_words = set(self._tokenize(text))
            if not doc_words:
                continue

            # Jaccard 相似度 + TF-IDF 近似
            intersection = query_words & doc_words
            union = query_words | doc_words
            jaccard = len(intersection) / len(union) if union else 0

            # 加权：长文档惩罚
            length_penalty = min(1.0, 200 / max(len(doc_words), 1))
            scores[doc_id] = jaccard * 0.7 + length_penalty * 0.3

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                **self.documents[doc_id],
                "score": score,
                "retriever": "vector",
                "rank": i + 1,
            }
            for i, (doc_id, score) in enumerate(ranked)
        ]

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合排序

        RRF_score(doc) = sum( 1 / (k + rank_i) )
        """
        rrf_scores: Dict[int, float] = {}
        doc_map: Dict[int, Dict] = {}

        # BM25 贡献
        for result in bm25_results:
            doc_id = result.get("doc_id", hash(result.get("text", "")))
            rank = result.get("rank", 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.bm25_weight / (self.rrf_k + rank)
            doc_map[doc_id] = result

        # Vector 贡献
        for result in vector_results:
            doc_id = result.get("doc_id", hash(result.get("text", "")))
            rank = result.get("rank", 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.vector_weight / (self.rrf_k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # 排序
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                **doc_map[doc_id],
                "rrf_score": score,
                "retriever": "hybrid_rrf",
            }
            for doc_id, score in ranked
        ]
