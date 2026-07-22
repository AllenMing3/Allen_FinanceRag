"""
KB 持久化工具 — 知识库读写的统一入口

将 services/persistence.py 的核心能力注册为 Tool，
让 Agent 和系统能力清单都能看到"知识库管理"这组能力。

设计原则:
- 本模块是 persistence 层的 tool 包装，不重复实现逻辑
- app_state / routers 仍可直接 import persistence（基础设施路径）
- Agent 通过 call_tool("kb_load") 等调用（编排路径）
- 未来换存储格式（JSON → 目录制），只改 persistence.py，tool 接口不变
"""
import logging
from typing import Dict, List, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


# ===================== Tool 实现函数 =====================

def kb_load() -> Dict:
    """加载知识库全部文档"""
    from financial_rag.services.persistence import load_kb, KB_PATH
    docs = load_kb()
    return {
        "count": len(docs),
        "path": KB_PATH,
        "docs": docs,
    }


def kb_save(docs: List[Dict]) -> Dict:
    """保存知识库（全量覆写 + 版本号递增）"""
    from financial_rag.services.persistence import save_kb, get_version
    save_kb(docs)
    return {
        "saved_count": len(docs),
        "version": get_version(),
    }


def kb_add(new_docs: List[Dict]) -> Dict:
    """增量添加文档到知识库（自动去重 + 分配 doc_id）"""
    from financial_rag.services.persistence import (
        load_kb, save_kb, assign_doc_ids, dedup_docs,
    )
    existing = load_kb()
    assign_doc_ids(existing)
    assign_doc_ids(new_docs)
    added = dedup_docs(existing, new_docs)
    if added:
        existing.extend(added)
        save_kb(existing)
    return {
        "total": len(existing),
        "added": len(added),
        "skipped_duplicate": len(new_docs) - len(added),
    }


def kb_remove(doc_ids: List[str]) -> Dict:
    """按 doc_id 删除知识库文档"""
    from financial_rag.services.persistence import load_kb, save_kb
    docs = load_kb()
    id_set = set(doc_ids)
    remaining = [d for d in docs if d.get("meta", {}).get("doc_id") not in id_set]
    removed_count = len(docs) - len(remaining)
    if removed_count > 0:
        save_kb(remaining)
    return {
        "removed": removed_count,
        "remaining": len(remaining),
    }


def kb_stats() -> Dict:
    """获取知识库统计信息"""
    from financial_rag.services.persistence import load_kb, load_stats, get_version
    docs = load_kb()
    stats = load_stats()

    # 计算长度分布
    lengths = [len(d.get("text", "")) for d in docs]
    sources = {}
    for d in docs:
        src = d.get("meta", {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    return {
        "doc_count": len(docs),
        "version": get_version(),
        "avg_length": sum(lengths) // max(len(lengths), 1),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "sources": sources,
        "total_analyses": stats.get("total_analyses", 0),
    }


# ===================== FunctionDef 注册列表 =====================

KB_TOOLS: List[FunctionDef] = [
    FunctionDef(
        name="kb_load",
        description="加载知识库全部文档。返回文档列表和数量。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        callback=kb_load,
        category="kb",
        tags=["知识库", "读取", "加载"],
    ),
    FunctionDef(
        name="kb_save",
        description="保存知识库（全量覆写）。传入完整文档列表，自动版本号递增。",
        parameters={
            "type": "object",
            "properties": {
                "docs": {
                    "type": "array",
                    "description": "完整的文档列表 [{text, meta}, ...]",
                    "items": {"type": "object"},
                },
            },
            "required": ["docs"],
        },
        callback=kb_save,
        category="kb",
        tags=["知识库", "保存", "写入"],
    ),
    FunctionDef(
        name="kb_add",
        description="增量添加文档到知识库。自动去重（基于 doc_id）和分配 ID。",
        parameters={
            "type": "object",
            "properties": {
                "new_docs": {
                    "type": "array",
                    "description": "要添加的新文档列表 [{text, meta}, ...]",
                    "items": {"type": "object"},
                },
            },
            "required": ["new_docs"],
        },
        callback=kb_add,
        category="kb",
        tags=["知识库", "添加", "增量"],
    ),
    FunctionDef(
        name="kb_remove",
        description="按 doc_id 删除知识库中的文档。",
        parameters={
            "type": "object",
            "properties": {
                "doc_ids": {
                    "type": "array",
                    "description": "要删除的文档 doc_id 列表",
                    "items": {"type": "string"},
                },
            },
            "required": ["doc_ids"],
        },
        callback=kb_remove,
        category="kb",
        tags=["知识库", "删除"],
    ),
    FunctionDef(
        name="kb_stats",
        description="获取知识库统计信息：文档数、版本、长度分布、来源分布。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        callback=kb_stats,
        category="kb",
        tags=["知识库", "统计", "状态"],
    ),
]
