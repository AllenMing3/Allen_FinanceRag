"""
Graph Tools — 知识图谱查询工具

让 Agent 通过 Function Calling 主动查询 LightRAG 知识图谱，
用于"公司A和公司B有什么关系"等图谱推理场景。

设计原则:
    1. 与 extraction_tools.py 对齐 — 闭包注入 adapter，FunctionDef 注册
    2. 返回结构化数据 — entities + relations，而非纯文本
    3. 容错设计 — adapter 不可用时返回空结果，不阻断 Pipeline

工具列表:
    - query_knowledge_graph: 查询实体关系图谱
    - get_graph_stats: 获取图谱统计概览
"""
import logging
from typing import Dict, Any, List

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


# ===================== Adapter 注入 (闭包模式) =====================

_adapter_ref = {"adapter": None}


def inject_graph_adapter(adapter):
    """注入 LightRAGAdapter 实例 — 由 create_financial_registry() 调用"""
    _adapter_ref["adapter"] = adapter


def _get_adapter():
    return _adapter_ref["adapter"]


# ===================== Tool 1: 查询知识图谱 =====================

def query_knowledge_graph(query: str, mode: str = "hybrid") -> Dict[str, Any]:
    """查询 AI/科技行业知识图谱，获取实体间的关系和上下文。

    适用场景:
    - 查询公司间的竞争/合作/投资关系
    - 查询某人在哪些公司任职
    - 查询某技术/产品的上下游关系
    - 查询政策对哪些公司/行业有影响

    支持四种查询模式:
    - local: 精确查询，聚焦直接相关的实体关系
    - global: 全局查询，捕捉间接关联和跨领域关系
    - hybrid: 混合模式（推荐），兼顾精确度和关联广度
    - mix: 最全面的查询，融合 local + global 结果

    Args:
        query: 自然语言查询（如"商汤科技和OpenAI有什么关系"）
        mode: 查询模式，可选 local/global/hybrid/mix，默认 hybrid
    """
    adapter = _get_adapter()
    if not adapter:
        return {
            "_confidence": "none",
            "_error": "知识图谱未初始化",
            "answer": "",
            "entities": [],
            "relations": [],
        }

    if not query or not query.strip():
        return {
            "_confidence": "none",
            "_error": "查询文本为空",
            "answer": "",
            "entities": [],
            "relations": [],
        }

    try:
        results = adapter.query(query.strip(), mode=mode)

        if not results:
            return {
                "_confidence": "low",
                "_source": "graph",
                "answer": "图谱中未找到相关信息",
                "entities": [],
                "relations": [],
                "graph_mode": mode,
            }

        # 取第一条结果（LightRAG 返回的是综合回答）
        item = results[0]
        return {
            "_confidence": "high",
            "_source": "graph",
            "answer": item.get("text", ""),
            "graph_mode": mode,
        }

    except Exception as e:
        logger.warning(f"[query_knowledge_graph] 查询失败: {e}")
        return {
            "_confidence": "none",
            "_error": f"图谱查询失败: {e}",
            "answer": "",
            "entities": [],
            "relations": [],
        }


# ===================== Tool 2: 图谱统计概览 =====================

def get_graph_stats() -> Dict[str, Any]:
    """获取知识图谱的统计概览信息。

    返回实体数量、关系数量、示例标签等，
    用于了解当前图谱的数据规模和覆盖范围。
    """
    adapter = _get_adapter()
    if not adapter:
        return {
            "_confidence": "none",
            "_error": "知识图谱未初始化",
            "initialized": False,
        }

    try:
        stats = adapter.get_graph_stats()
        stats["_confidence"] = "high" if stats.get("initialized") else "none"
        stats["_source"] = "graph"
        return stats
    except Exception as e:
        logger.warning(f"[get_graph_stats] 获取统计失败: {e}")
        return {"_confidence": "none", "_error": f"获取统计失败: {e}"}


# ===================== FunctionDef 列表 =====================

GRAPH_TOOLS: List[FunctionDef] = [
    FunctionDef(
        name="query_knowledge_graph",
        description="查询 AI/科技行业知识图谱，获取实体间的关系和上下文。"
                    "适合查询公司间的竞争/合作/投资关系、人物任职关系、技术上下游、政策影响等。"
                    "支持 local（精确）/global（全局）/hybrid（混合，推荐）/mix（最全面）四种模式。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言查询（如'商汤科技和OpenAI有什么关系'）",
                },
                "mode": {
                    "type": "string",
                    "description": "查询模式: local/global/hybrid/mix，默认 hybrid",
                    "enum": ["local", "global", "hybrid", "mix"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
        callback=query_knowledge_graph,
        category="retrieval",
        tags=["知识图谱", "实体关系", "图谱查询", "LightRAG"],
    ),
    FunctionDef(
        name="get_graph_stats",
        description="获取知识图谱的统计概览（实体数量、关系数量、示例标签等）。"
                    "用于了解当前图谱的数据规模和覆盖范围。",
        parameters={
            "type": "object",
            "properties": {},
        },
        callback=get_graph_stats,
        category="retrieval",
        tags=["知识图谱", "统计", "概览"],
    ),
]
