"""
调度工具 — 供 CoordinatorAgent 调用的路由与调度能力

工具:
- classify_query_intent: 识别查询意图 (kline/event_impact/report/news/general)
- select_agent_chain: 根据意图选择 Agent 执行链
"""
import logging
from typing import Dict, Any, List, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


# ===================== 工具实现 =====================


def classify_query_intent(query: str) -> Dict:
    """识别用户查询意图，返回意图类型和置信度。

    意图类型:
    - kline: K线技术分析 (走势、MACD、RSI 等)
    - event_impact: 事件影响分析 (利好利空、事件冲击)
    - report: 财报分析 (营收、利润、年报)
    - news: 新闻综合 (行业动态、最新资讯)
    - general: 通用查询

    Args:
        query: 用户自然语言查询
    """
    from financial_rag.core.agent_router import create_agent_router

    router = create_agent_router()
    intent = router.classify(query)

    return {
        "intent": intent.name,
        "confidence": intent.confidence,
        "matched_keywords": intent.matched_keywords,
    }


def select_agent_chain(intent: str, confidence: float = 0.5) -> Dict:
    """根据意图选择 Agent 执行链。

    链由 AgentRouter 动态定义，典型链:
    - kline → [KLineAgent, ReportAgent, ScoringAgent]
    - event_impact → [EventImpactAgent, ReportAgent, ScoringAgent]
    - report → [IngestionAgent, ExtractionAgent, ReportAgent, ScoringAgent]
    - news → [IngestionAgent, ReportAgent, ScoringAgent]
    - general → [IngestionAgent, ExtractionAgent, ReportAgent, ScoringAgent]

    Args:
        intent: 意图类型 (来自 classify_query_intent)
        confidence: 路由置信度 (0-1)
    """
    from financial_rag.core.agent_router import create_agent_router

    router = create_agent_router()
    intent_map = router.get_intent_map()

    chain = intent_map.get(intent, intent_map.get("_default", []))

    # 低置信度时，添加 IngestionAgent 作为数据预处理
    if confidence < 0.4 and "IngestionAgent" not in chain:
        chain = ["IngestionAgent"] + chain

    return {
        "intent": intent,
        "confidence": confidence,
        "agent_chain": chain,
        "chain_description": " → ".join(chain),
    }


# ===================== FunctionDef 定义 =====================

CLASSIFY_INTENT_TOOL = FunctionDef(
    name="classify_query_intent",
    description="识别用户查询意图，分类为 kline/event_impact/report/news/general 五种类型。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户自然语言查询"},
        },
        "required": ["query"],
    },
    callback=classify_query_intent,
    category="analysis",
    tags=["路由", "意图", "分类"],
)

SELECT_CHAIN_TOOL = FunctionDef(
    name="select_agent_chain",
    description="根据查询意图选择最优 Agent 执行链。",
    parameters={
        "type": "object",
        "properties": {
            "intent": {"type": "string", "description": "意图类型 (kline/event_impact/report/news/general)"},
            "confidence": {"type": "number", "description": "置信度 0-1", "default": 0.5},
        },
        "required": ["intent"],
    },
    callback=select_agent_chain,
    category="analysis",
    tags=["路由", "调度", "Agent链"],
)

COORDINATOR_TOOLS = [CLASSIFY_INTENT_TOOL, SELECT_CHAIN_TOOL]
