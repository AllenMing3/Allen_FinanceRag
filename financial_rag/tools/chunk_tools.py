"""
文档切分工具 — 将长文档切分为检索友好的 chunks

将 chunker.py 的切分能力注册为 Tool，并支持按 doc_type 选择策略。
未来可扩展语义切分、按章节切分等策略，外部通过 strategy 参数控制。

策略:
- "default": 当前行为（500字符 + 50重叠，递归按段落/句子边界）
- "none": 不切分（短文 / 新闻 < chunk_size 时自动跳过）
- "paragraph": 按段落切分（\n\n 边界），适合长文/报告

用法:
    # Agent 调用
    result = self.call_tool("chunk_document", text=long_text, strategy="paragraph")

    # 未来: retriever 内部也可走这个统一入口
"""
import logging
from typing import Dict, List, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


# ===================== 策略配置 =====================

STRATEGY_CONFIGS = {
    "default": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "min_chunk_size": 50,
    },
    "paragraph": {
        "chunk_size": 1500,
        "chunk_overlap": 100,
        "min_chunk_size": 100,
    },
    "none": {
        "chunk_size": 999999,  #  effectively no split
        "chunk_overlap": 0,
        "min_chunk_size": 1,
    },
}


# ===================== Tool 实现函数 =====================

def chunk_document(
    text: str,
    strategy: str = "default",
    meta: Optional[Dict] = None,
    source_id: str = "",
) -> Dict:
    """将文本按指定策略切分为 chunks。

    Args:
        text: 原始文本
        strategy: 切分策略 (default / paragraph / none)
        meta: 附加元数据（会传递给每个 chunk）
        source_id: 来源文档 ID（用于追溯）

    Returns:
        {"chunks": [{text, meta}, ...], "chunk_count": N, "strategy": "..."}
    """
    from financial_rag.retrievers.chunker import TextChunker

    cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["default"])

    chunker = TextChunker(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        min_chunk_size=cfg["min_chunk_size"],
    )

    chunks = chunker.split(text, meta=meta or {}, source_id=source_id)

    return {
        "chunks": chunks,
        "chunk_count": len(chunks),
        "strategy": strategy,
        "config": cfg,
        "original_length": len(text),
    }


def chunk_documents_batch(
    documents: List[Dict],
    strategy: str = "default",
) -> Dict:
    """批量切分文档列表。

    Args:
        documents: [{"text": "...", "meta": {...}}, ...]
        strategy: 切分策略

    Returns:
        {"chunks": [...], "input_count": N, "output_count": M, "strategy": "..."}
    """
    from financial_rag.retrievers.chunker import TextChunker

    cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["default"])

    chunker = TextChunker(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        min_chunk_size=cfg["min_chunk_size"],
    )

    all_chunks = chunker.split_documents(documents)

    return {
        "chunks": all_chunks,
        "input_count": len(documents),
        "output_count": len(all_chunks),
        "strategy": strategy,
    }


# ===================== FunctionDef 注册列表 =====================

CHUNK_TOOLS: List[FunctionDef] = [
    FunctionDef(
        name="chunk_document",
        description="将长文本按指定策略切分为检索友好的 chunks。"
                    "策略: default(500字/50重叠), paragraph(1500字/100重叠,适合报告), none(不切分)。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要切分的原始文本"},
                "strategy": {
                    "type": "string",
                    "description": "切分策略: default / paragraph / none",
                    "default": "default",
                },
                "meta": {
                    "type": "object",
                    "description": "附加元数据，会传递给每个 chunk",
                },
                "source_id": {
                    "type": "string",
                    "description": "来源文档 ID（用于追溯）",
                    "default": "",
                },
            },
            "required": ["text"],
        },
        callback=chunk_document,
        category="kb",
        tags=["切分", "chunk", "文档处理"],
    ),
    FunctionDef(
        name="chunk_documents_batch",
        description="批量切分文档列表。传入 [{text, meta}, ...] 返回所有 chunks。",
        parameters={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "文档列表 [{text, meta}, ...]",
                    "items": {"type": "object"},
                },
                "strategy": {
                    "type": "string",
                    "description": "切分策略: default / paragraph / none",
                    "default": "default",
                },
            },
            "required": ["documents"],
        },
        callback=chunk_documents_batch,
        category="kb",
        tags=["切分", "chunk", "批量"],
    ),
]
