"""
TextChunker — 文档切分器

将长文档切分为适合检索的语义块（chunks），提升检索精度。

策略:
- 按字符数切分，支持中文和英文
- 优先在段落/句子边界切分，避免截断语义
- 支持 chunk 重叠，保留上下文连贯性

用法:
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(text, meta={"source": "news"})
"""
from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """
    文档切分器 — 将长文档切分为检索友好的 chunks

    设计原则:
    1. 优先在段落边界切分（\n\n）
    2. 其次在句子边界切分（。！？.!?）
    3. 最后硬切分（按字符数）
    4. chunk 间有重叠，保留上下文

    Args:
        chunk_size: 每个 chunk 的最大字符数（默认 500）
        chunk_overlap: 相邻 chunk 的重叠字符数（默认 50）
        min_chunk_size: 最小 chunk 字符数，低于此值的合并到前一个（默认 50）
        separators: 切分优先级分隔符列表
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ","]

    def split(
        self,
        text: str,
        meta: Optional[Dict] = None,
        source_id: str = "",
    ) -> List[Dict]:
        """
        将文本切分为多个 chunk

        Args:
            text: 原始文本
            meta: 附加到每个 chunk 的元数据
            source_id: 来源文档 ID（用于追溯）

        Returns:
            [{"text": "...", "meta": {..., "chunk_id": 0, "source_id": "..."}, ...]
        """
        if not text or not text.strip():
            logger.warning(f"TextChunker.split: empty text from source_id={source_id or 'unknown'}")
            return []

        meta = meta or {}

        # 短文直接返回，无需切分
        if len(text) <= self.chunk_size:
            return [{
                "text": text.strip(),
                "meta": {**meta, "chunk_id": 0, "chunk_count": 1, "source_id": source_id},
            }]

        # 递归切分
        raw_chunks = self._split_recursive(text, self.separators)

        # 合并过小的 chunks
        merged = self._merge_small_chunks(raw_chunks)

        # 添加重叠
        overlapped = self._add_overlap(merged)

        # 构建结果
        results = []
        empty_dropped = 0
        for i, chunk_text in enumerate(overlapped):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                empty_dropped += 1
                continue
            # 查找 chunk 在原文中的位置
            search_anchor = chunk_text[:20] if len(chunk_text) > 20 else chunk_text
            pos = text.find(search_anchor)
            chunk_meta = {
                **meta,
                "chunk_id": i,
                "chunk_count": len(overlapped),
                "source_id": source_id,
                "chunk_start": pos if pos >= 0 else -1,
            }
            if pos < 0:
                logger.debug(
                    f"TextChunker: chunk_start not found in source_id={source_id}, "
                    f"chunk_id={i} (overlap may have shifted position)"
                )
            results.append({"text": chunk_text, "meta": chunk_meta})

        if empty_dropped > 0:
            logger.warning(
                f"TextChunker: dropped {empty_dropped} empty chunks from source_id={source_id}"
            )

        return results

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        批量切分文档列表

        Args:
            documents: [{"text": "...", "meta": {...}}, ...]

        Returns:
            切分后的 chunk 列表
        """
        all_chunks = []
        empty_docs = 0
        for doc_id, doc in enumerate(documents):
            text = doc.get("text", "")
            meta = doc.get("meta", {})
            source_id = meta.get("url") or meta.get("title") or f"doc_{doc_id}"
            if not text or not text.strip():
                empty_docs += 1
                logger.warning(
                    f"TextChunker.split_documents: doc_{doc_id} has empty text, "
                    f"skipping (source={source_id})"
                )
                continue
            chunks = self.split(text, meta=meta, source_id=source_id)
            all_chunks.extend(chunks)

        if empty_docs > 0:
            logger.warning(
                f"TextChunker: {empty_docs}/{len(documents)} documents had empty text"
            )

        logger.info(
            f"TextChunker: {len(documents)} 篇文档 → {len(all_chunks)} 个 chunks"
            f" (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归使用分隔符切分文本，直到每个 chunk 小于 chunk_size"""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # 硬切分
            return self._hard_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)

        # 如果切分后只有一个 part，说明这个分隔符没用，尝试下一个
        if len(parts) <= 1:
            return self._split_recursive(text, remaining_seps)

        # 贪心合并 parts 到 chunk_size 以内
        chunks = []
        current = ""
        for part in parts:
            # 加上分隔符（除了第一个）
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 如果单个 part 就超过 chunk_size，递归用下一级分隔符
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_recursive(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """按字符数硬切分"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的 chunks 到前一个"""
        if not chunks:
            return []

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk) < self.min_chunk_size and merged:
                # 合并到前一个
                merged[-1] = merged[-1] + chunk
            else:
                merged.append(chunk)

        return merged

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """在相邻 chunks 间添加重叠"""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            # 从前一个 chunk 末尾取 overlap 长度的文本
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            result.append(overlap_text + chunks[i])

        return result
