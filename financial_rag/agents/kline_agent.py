"""
KLineAgent — K 线技术分析 Agent

功能:
- 从 context 解析股票代码
- 委托 analyze_kline 工具获取 K 线数据 + 计算技术指标
- 委托 generate_kline_analysis 工具生成 LLM 分析摘要
- 返回结构化分析结果

Agent 只做编排决策，所有数据获取和计算委托给 tools。
"""
import re
import logging
from typing import Dict, Any

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult
from financial_rag.tools.kline_tools import STOCK_MAP, KLINE_ANALYSIS_SYSTEM, KLINE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


# 重导出供外部使用（向后兼容）
__all__ = ["KLineAgent", "STOCK_MAP", "KLINE_ANALYSIS_SYSTEM", "KLINE_ANALYSIS_PROMPT"]


class KLineAgent(BaseAgent):
    """
    K 线技术分析 Agent

    轻量级编排者:
    1. 从 context 解析股票代码
    2. 调用 analyze_kline 工具获取数据 + 指标
    3. 调用 generate_kline_analysis 工具生成分析
    4. 返回结构化结果
    """

    def __init__(self):
        super().__init__(
            name="KLineAgent",
            description="K 线技术分析：获取行情数据、计算技术指标、LLM 生成分析报告",
        )

    def can_handle(self, context: AgentContext) -> bool:
        """K 线技术分析类查询：有股票代码、走势关键词、或 router 已标记意图"""
        intent = context.metadata.get("intent", "")
        if intent == "kline":
            return True
        raw = context.raw_input
        # 有股票代码
        if context.metadata.get("ts_code") or context.metadata.get("stock_code"):
            return True
        # K线关键词
        kline_kws = ["K线", "k线", "走势", "技术分析", "MACD", "RSI", "均线", "KDJ",
                     "支撑位", "压力位", "趋势", "布林", "多头", "空头", "震荡",
                     "涨跌", "收盘", "开盘", "成交量", "日K", "周K", "月K"]
        if any(kw in raw for kw in kline_kws):
            return True
        # 股票名称映射
        for keyword in STOCK_MAP:
            if keyword in raw:
                return True
        return False

    def process(self, context: AgentContext) -> AgentResult:
        """
        处理 K 线分析请求

        全部业务逻辑委托给工具:
        - analyze_kline: 数据获取 + 指标计算
        - generate_kline_analysis: LLM 分析生成
        """
        raw_input = context.raw_input
        ts_code = context.metadata.get("ts_code", "")
        name = context.metadata.get("name", "")
        days = context.metadata.get("days", 60)
        period = context.metadata.get("period", "daily")

        # 如果没有 ts_code，尝试从查询中提取
        if not ts_code:
            ts_code, name = self._extract_stock_code(raw_input)

        if not ts_code:
            return AgentResult(
                success=False,
                message="无法识别股票代码，请提供具体的股票名称或代码",
                context_updates={"final_answer": "无法识别股票代码，请提供具体的股票名称或代码（如：贵州茅台、600519）"},
            )

        # Step 1: 委托工具获取 K 线数据 + 计算指标
        kline_data = self.call_tool(
            "analyze_kline", ts_code=ts_code, days=days, period=period
        )

        if "error" in kline_data:
            return AgentResult(
                success=False,
                message=kline_data["error"],
                context_updates={"final_answer": f"{kline_data['error']}，请检查股票代码是否正确或 Tushare Token 是否已配置"},
            )

        stats = kline_data.get("stats", {})
        indicators = kline_data.get("indicators", {})

        # Step 2: 委托工具生成 LLM 分析
        analysis_result = self.call_tool(
            "generate_kline_analysis",
            ts_code=ts_code,
            name=name or ts_code,
            stats=stats,
            indicators=indicators,
        )

        analysis = analysis_result.get("analysis", "分析生成失败")

        # Step 3: 组装结果（纯编排，无业务逻辑）
        result_data = {
            "ts_code": ts_code,
            "name": name or ts_code,
            "days": days,
            "data_points": kline_data.get("data_points", 0),
            "stats": stats,
            "indicators": indicators,
            "analysis": analysis,
        }

        return AgentResult(
            success=True,
            message=f"K 线分析完成: {name or ts_code} ({kline_data.get('data_points', 0)} 个交易日)",
            data=result_data,
            context_updates={
                "final_answer": analysis,
                "intermediate_findings": [{
                    "stage": "kline_analysis",
                    "ts_code": ts_code,
                    "data_points": kline_data.get("data_points", 0),
                    "stats": stats,
                    "indicators": indicators,
                }],
            }
        )

    def _extract_stock_code(self, query: str) -> tuple:
        """从用户查询中提取股票代码（纯路由辅助，非业务逻辑）"""
        # 正则匹配 6 位代码
        code_match = re.search(r'(\d{6})', query)
        if code_match:
            code = code_match.group(1)
            if code.startswith("6"):
                ts_code = f"{code}.SH"
            else:
                ts_code = f"{code}.SZ"
            return ts_code, code

        # 关键词映射
        for keyword, (ts_code, name) in STOCK_MAP.items():
            if keyword in query:
                return ts_code, name

        return "", ""
