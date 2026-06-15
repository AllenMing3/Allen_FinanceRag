"""
EventImpactAgent — 事件影响分析 Agent（V1）

将新闻事件与 K 线数据做映射，判断利好/利空及影响因子。

V1 能力:
- 给定日期 → 拉取当日事件 → LLM 评估影响（利好/利空/中性 + 影响因子 0-10）
- 可选传入股票代码 → 同时拉取 K 线数据辅助判断

Agent 只做编排决策，所有数据获取和分析委托给 tools:
- fetch_date_events: 获取某日事件
- fetch_kline_context: 获取 K 线上下文
- assess_event_impact: LLM 事件影响评估
"""
import logging
from typing import Dict, Any

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult

logger = logging.getLogger(__name__)


class EventImpactAgent(BaseAgent):
    """
    事件影响分析 Agent

    轻量级决策者:
    1. 从 context 解析 date + stock_code
    2. 调用 fetch_date_events 获取事件
    3. 可选调用 fetch_kline_context 获取K线
    4. 调用 assess_event_impact 做影响评估
    5. 返回结构化结果
    """

    def __init__(self):
        super().__init__(
            name="EventImpactAgent",
            description="事件影响分析: 日期事件 → 利好/利空判断 → 影响因子评估",
        )

    def can_handle(self, context: AgentContext) -> bool:
        """事件影响类查询：有日期、事件关键词、或 router 已标记意图"""
        intent = context.metadata.get("intent", "")
        if intent == "event_impact":
            return True
        raw = context.raw_input.lower()
        # 有日期模式
        if self._extract_date(raw):
            return True
        # 事件类关键词
        event_kws = ["利好", "利空", "事件", "大事", "发生", "冲击", "影响", "并购", "收购",
                     "重组", "涨停", "跌停", "暴涨", "暴跌"]
        return any(kw in raw for kw in event_kws)

    def process(self, context: AgentContext) -> AgentResult:
        """
        处理事件影响分析请求

        context.raw_input: 用户查询（如 "2024年6月1日发生了什么"、"茅台最近有什么大事"）
        context.metadata 可包含:
            - date: 日期 'YYYY-MM-DD'
            - stock_code: Tushare 代码 (如 '600519.SH')
            - stock_name: 股票名称 (如 '贵州茅台')
            - keyword: 关键词过滤
        """
        raw_input = context.raw_input
        date = context.metadata.get("date", "")
        stock_code = context.metadata.get("stock_code", "")
        stock_name = context.metadata.get("stock_name", "")
        keyword = context.metadata.get("keyword", "")

        # 从查询中提取日期（如果 metadata 没有）
        if not date:
            date = self._extract_date(raw_input)

        # 从查询中提取股票（如果 metadata 没有）
        if not stock_code:
            stock_code, stock_name = self._extract_stock(raw_input)

        if not date and not keyword:
            return AgentResult(
                success=False,
                message="无法确定分析目标，请提供日期或关键词",
                data={"error": "missing_params"},
                context_updates={
                    "final_answer": "请提供要分析的日期（如 2024-06-01）或关键词（如 AI芯片）"
                },
            )

        # ---- Step 1: 获取事件 ----
        events_data = self._call_fetch_events(date, keyword)
        events = events_data.get("events", [])

        if not events:
            return AgentResult(
                success=False,
                message=f"未找到 {date or keyword} 相关事件",
                data={"date": date, "events": []},
                context_updates={
                    "final_answer": f"未找到 {date or keyword} 相关的事件/新闻，可能当日无重大消息"
                },
            )

        # ---- Step 2: 可选获取 K 线 ----
        kline_context = None
        if stock_code and self._registry:
            try:
                kline_context = self.call_tool(
                    "fetch_kline_context",
                    stock_code=stock_code,
                    date=date,
                    window_days=10,
                )
            except Exception as e:
                logger.warning(f"K线获取失败: {e}")

        # ---- Step 3: 影响评估 ----
        assessment = self._call_assess_impact(events, kline_context, stock_name)

        # ---- Step 4: 构建结果 ----
        result_data = {
            "date": date,
            "keyword": keyword,
            "stock_code": stock_code,
            "stock_name": stock_name,
            "event_count": len(events),
            "events": events,
            "kline_context": kline_context,
            "assessment": assessment,
        }

        # 构建可读的最终回答
        answer = self._format_answer(result_data)

        return AgentResult(
            success=True,
            message=f"事件影响分析完成: {len(events)} 个事件",
            data=result_data,
            context_updates={
                "final_answer": answer,
                "intermediate_findings": [{
                    "stage": "event_impact",
                    "event_count": len(events),
                    "assessment": assessment,
                }],
            },
        )

    def _call_fetch_events(self, date: str, keyword: str) -> Dict:
        """调用 fetch_date_events 工具"""
        try:
            return self.call_tool(
                "fetch_date_events", date=date, keyword=keyword
            )
        except Exception as e:
            logger.warning(f"事件获取失败: {e}")
            return {"events": [], "error": str(e)}

    def _call_assess_impact(
        self, events: list, kline_context: Dict, stock_name: str
    ) -> Dict:
        """调用 assess_event_impact 工具"""
        try:
            return self.call_tool(
                "assess_event_impact",
                events=events,
                kline_context=kline_context,
                stock_name=stock_name,
            )
        except Exception as e:
            logger.warning(f"影响评估失败: {e}")
            return {"assessments": [], "overall_label": "未知", "overall_factor": 0, "error": str(e)}

    def _format_answer(self, data: Dict) -> str:
        """将结构化数据格式化为可读文本"""
        lines = []
        date = data.get("date", "未知日期")
        stock = data.get("stock_name", "") or data.get("stock_code", "")
        assessment = data.get("assessment", {})

        lines.append(f"## 事件影响分析: {date}")
        if stock:
            lines.append(f"标的: {stock}")
        lines.append("")

        # 事件概览
        events = data.get("events", [])
        lines.append(f"### 当日事件 ({len(events)} 条)")
        for i, e in enumerate(events[:5], 1):
            lines.append(f"{i}. {e.get('title', '')[:60]}")
        if len(events) > 5:
            lines.append(f"   ... 还有 {len(events) - 5} 条")
        lines.append("")

        # 影响评估
        assessments = assessment.get("assessments", [])
        if assessments:
            lines.append("### 影响评估")
            for a in assessments:
                icon = "📈" if a.get("impact") == "bullish" else (
                    "📉" if a.get("impact") == "bearish" else "➡️"
                )
                factor = a.get("impact_factor", 0)
                bar = "█" * factor + "░" * (10 - factor)
                lines.append(
                    f"{icon} {a.get('event', '')[:40]} "
                    f"[{a.get('impact_label', '?')}] "
                    f"影响力: {bar} ({factor}/10)"
                )
                if a.get("reasoning"):
                    lines.append(f"   → {a['reasoning']}")
            lines.append("")

        # 综合判断
        lines.append("### 综合判断")
        overall = assessment.get("overall_label", "未知")
        overall_factor = assessment.get("overall_factor", 0)
        lines.append(f"**{overall}** (综合影响力: {overall_factor}/10)")
        if assessment.get("summary"):
            lines.append(f"> {assessment['summary']}")

        # K 线补充
        kline = data.get("kline_context")
        if kline and "error" not in kline:
            lines.append("")
            lines.append("### K线参考")
            lines.append(f"- 事件前平均涨跌: {kline.get('before_avg_change_pct', 'N/A')}%")
            lines.append(f"- 事件后平均涨跌: {kline.get('after_avg_change_pct', 'N/A')}%")

        lines.append("")
        lines.append("*以上分析仅供参考，不构成投资建议。*")

        return "\n".join(lines)

    def _extract_date(self, query: str) -> str:
        """从查询中提取日期"""
        import re
        # 匹配 YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD
        patterns = [
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
            r'(\d{4})(\d{2})(\d{2})',
        ]
        for pat in patterns:
            m = re.search(pat, query)
            if m:
                y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
                return f"{y}-{mo}-{d}"
        return ""

    def _extract_stock(self, query: str) -> tuple:
        """从查询中提取股票代码"""
        import re
        from financial_rag.tools.kline_tools import STOCK_MAP

        # 关键词匹配
        for keyword, (ts_code, name) in STOCK_MAP.items():
            if keyword in query:
                return ts_code, name

        # 6位代码匹配
        m = re.search(r'(\d{6})', query)
        if m:
            code = m.group(1)
            if code.startswith("6"):
                return f"{code}.SH", code
            return f"{code}.SZ", code

        return "", ""
