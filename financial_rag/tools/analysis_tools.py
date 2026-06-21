"""
深度分析工具 — 封装 services/analysis.py 供 Agent 链调用

解决 agent 链和 service 层深度断裂问题:
- analyze_news_deep: 封装 analyze_news_text，返回结构化多维分析
- analyze_topic_deep: 封装 analyze_topic_research，返回结构化话题调研
"""
import logging
from typing import Dict, Any, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)

# 依赖注入引用
_analysis_refs = {"llm": None, "retriever": None, "kb_built": False}


def inject_analysis_deps(llm=None, retriever=None, kb_built: bool = False):
    """注入 LLM 和 Retriever 供深度分析工具使用"""
    _analysis_refs["llm"] = llm
    _analysis_refs["retriever"] = retriever
    _analysis_refs["kb_built"] = kb_built


def analyze_news_deep(
    text: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """对新闻文本进行深度结构化分析: 多维影响评估 + 关键信号 + 风险提示。

    Args:
        text: 新闻文本内容
        query: 用户查询（可选，用于聚焦分析方向）
    """
    if not text:
        return {"error": "新闻文本为空", "structured": {}}

    from financial_rag.services.analysis import analyze_news_text

    try:
        result = analyze_news_text(
            text,
            query=query,
            llm=_analysis_refs.get("llm"),
            retriever=_analysis_refs.get("retriever"),
            kb_built=_analysis_refs.get("kb_built", False),
        )
        return result
    except Exception as e:
        logger.error(f"[analyze_news_deep] 失败: {e}")
        return {"error": str(e), "structured": {}, "analysis": f"分析失败: {e}"}


def analyze_topic_deep(
    topic: str = "",
    max_news: int = 20,
) -> Dict[str, Any]:
    """对指定话题进行深度调研: 子话题聚类 + 关键参与者 + 情绪趋势 + 反向信号。

    Args:
        topic: 调研话题关键词
        max_news: 最大新闻抓取数
    """
    if not topic:
        return {"error": "话题为空", "structured": {}}

    from financial_rag.services.analysis import analyze_topic_research

    try:
        result = analyze_topic_research(
            topic,
            max_news=max_news,
            llm=_analysis_refs.get("llm"),
            retriever=_analysis_refs.get("retriever"),
            kb_built=_analysis_refs.get("kb_built", False),
        )
        return result
    except Exception as e:
        logger.error(f"[analyze_topic_deep] 失败: {e}")
        return {"error": str(e), "structured": {}, "analysis": f"调研失败: {e}"}


# 工具定义
ANALYSIS_TOOLS = [
    FunctionDef(
        name="analyze_news_deep",
        description="对新闻文本进行深度结构化分析，返回多维影响评估(行业/公司/技术/市场)、关键信号(含严重度)、风险提示、后续关注。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "新闻文本内容"},
                "query": {"type": "string", "description": "用户查询(可选)", "default": ""},
            },
            "required": ["text"],
        },
        callback=analyze_news_deep,
        category="analysis",
        tags=["新闻", "深度分析", "结构化", "影响评估"],
    ),
    FunctionDef(
        name="analyze_topic_deep",
        description="对指定话题进行深度调研，返回子话题聚类、关键参与者、情绪趋势、反向信号、投资启示。",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "调研话题关键词"},
                "max_news": {"type": "integer", "description": "最大新闻抓取数", "default": 20},
            },
            "required": ["topic"],
        },
        callback=analyze_topic_deep,
        category="analysis",
        tags=["话题", "调研", "结构化", "趋势"],
    ),
]
