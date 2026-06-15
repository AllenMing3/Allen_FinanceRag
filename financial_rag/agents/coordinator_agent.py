"""
CoordinatorAgent — 智能调度 Agent

职责:
- 分析查询意图
- 选择最优 Agent 执行链
- 将路由决策注入 AgentContext 供后续 Agent 使用

Agent 只做编排决策，所有分类和路由逻辑委托给 tools:
- classify_query_intent: 意图分类
- select_agent_chain: Agent 链选择
"""
import logging
from typing import Dict, Any

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    智能调度 Agent

    轻量级编排者:
    1. 调用 classify_query_intent 工具分类查询
    2. 调用 select_agent_chain 工具选择执行链
    3. 将意图和链信息注入 context 供后续 Agent 使用
    """

    def __init__(self):
        super().__init__(
            name="CoordinatorAgent",
            description="智能调度: 意图分类 + Agent 链选择 + 路由决策",
        )

    def can_handle(self, context: AgentContext) -> bool:
        """始终可以处理（作为链中第一个 Agent）"""
        return True

    def process(self, context: AgentContext) -> AgentResult:
        """执行智能调度 — 全部委托给工具"""
        query = context.raw_input

        # Step 1: 委托工具 — 分类查询意图
        intent_result = self.call_tool("classify_query_intent", query=query)
        intent = intent_result.get("intent", "general")
        confidence = intent_result.get("confidence", 0.5)

        # Step 2: 委托工具 — 选择 Agent 链
        chain_result = self.call_tool(
            "select_agent_chain", intent=intent, confidence=confidence
        )
        agent_chain = chain_result.get("agent_chain", [])

        # Step 3: 注入路由元数据到 context（供后续 Agent 的 can_handle 使用）
        # 提取元数据（日期、股票代码等）
        routing_metadata = self._extract_metadata(query)
        routing_metadata["intent"] = intent
        routing_metadata["confidence"] = confidence
        routing_metadata["agent_chain"] = agent_chain

        return AgentResult(
            success=True,
            message=f"调度完成: {intent} → {chain_result.get('chain_description', '')}",
            data={
                "intent": intent,
                "confidence": confidence,
                "agent_chain": agent_chain,
                "matched_keywords": intent_result.get("matched_keywords", []),
            },
            context_updates={
                "metadata": routing_metadata,
                "intermediate_findings": [{
                    "stage": "coordination",
                    "intent": intent,
                    "confidence": confidence,
                    "agent_chain": agent_chain,
                }],
            },
        )

    def _extract_metadata(self, query: str) -> Dict:
        """从查询中提取路由辅助元数据（纯路由辅助，非业务逻辑）"""
        import re
        meta = {}

        # 日期提取
        patterns = [
            r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?',
            r'(\d{4})(\d{2})(\d{2})',
        ]
        for pat in patterns:
            m = re.search(pat, query)
            if m:
                y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
                meta["date"] = f"{y}-{mo}-{d}"
                break

        # 股票代码提取
        try:
            from financial_rag.tools.kline_tools import STOCK_MAP
            for keyword, (ts_code, name) in STOCK_MAP.items():
                if keyword in query:
                    meta["stock_code"] = ts_code
                    meta["ts_code"] = ts_code
                    meta["stock_name"] = name
                    break
        except ImportError:
            pass

        if "stock_code" not in meta:
            m = re.search(r'(\d{6})', query)
            if m:
                code = m.group(1)
                ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
                meta["stock_code"] = ts_code
                meta["ts_code"] = ts_code

        return meta
