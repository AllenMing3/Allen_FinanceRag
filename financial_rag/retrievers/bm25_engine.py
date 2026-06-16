"""
BM25 检索引擎 — BM25Okapi 封装

职责:
- 构建/重建 BM25Okapi 索引
- 关键词检索，返回带 rank 的排序结果
- 分词回退（无 jieba 时）
"""
import re
import logging
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Engine:
    """BM25Okapi 封装 — 构建索引 + 检索"""

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: 分词函数 callable(text) -> List[str]，None 则用回退分词
        """
        self._tokenizer = tokenizer
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, documents: List[Dict]):
        """从文档列表构建 BM25 索引"""
        self._corpus_tokens = [
            self.tokenize(doc.get("text", "")) for doc in documents
        ]
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def search(self, documents: List[Dict], query: str, top_k: int,
               query_tokens: List[str] = None) -> List[Dict]:
        """BM25 检索，返回带 score + rank 的结果"""
        if not documents or self._bm25 is None:
            return []
        query_terms = query_tokens if query_tokens is not None else self.tokenize(query)
        if not query_terms:
            return []

        scores = self._bm25.get_scores(query_terms)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            {
                **documents[doc_id],
                "score": float(scores[doc_id]),
                "retriever": "bm25",
                "rank": rank + 1,
            }
            for rank, doc_id in enumerate(ranked_indices)
            if scores[doc_id] > 0
        ]

    def tokenize(self, text: str) -> List[str]:
        """分词 — 优先注入的分词器，否则回退到正则"""
        if self._tokenizer is not None:
            try:
                return self._tokenizer(text)
            except Exception as e:
                logger.warning(f"BM25Engine: tokenizer failed, falling back to regex: {e}")
        return self._fallback_tokenize(text)

    def clear(self):
        self._bm25 = None
        self._corpus_tokens = []

    @staticmethod
    def _fallback_tokenize(text: str) -> List[str]:
        """回退分词: 中文双字滑窗 + 英文按单词"""
        raw = re.findall(
            r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+(?:\.\d+)?[%％]?', text.lower()
        )
        tokens = []
        for seg in raw:
            if re.match(r'^[\u4e00-\u9fff]+$', seg) and len(seg) > 1:
                for j in range(len(seg) - 1):
                    tokens.append(seg[j:j + 2])
                if len(seg) <= 6:
                    tokens.append(seg)
            else:
                tokens.append(seg)
        return tokens
