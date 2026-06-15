"""
ReportAgent — LLM 驱动的新闻综合分析 + 引用报告

功能:
- 将 parsed_data (KB 文档) + extracted_features (指标/实体) 综合为上下文
- 委托 synthesize_report 工具生成结构化分析报告
- 渲染为带引用的 Markdown 报告

Agent 只做编排决策，LLM 报告生成委托给 synthesize_report 工具。
"""
import json
import re
from typing import Dict, Any, List, Optional

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult


class ReportAgent(BaseAgent):
    """
    Agent: 新闻综合分析 + 报告生成

    轻量级编排者:
    1. 从 context 获取 parsed_data + extracted_features
    2. 构建 source 列表
    3. 委托 synthesize_report 工具生成报告
    4. 渲染 Markdown
    """

    def __init__(self):
        super().__init__(
            name="ReportAgent",
            description="LLM 驱动的新闻综合分析 + 引用报告"
        )

    def can_handle(self, context: AgentContext) -> bool:
        """需要 parsed_data / extracted_features / specialist 中间结果"""
        if context.parsed_data or context.extracted_features:
            return True
        if context.intermediate_findings:
            return True
        if context.final_answer:
            return True
        return False

    def process(self, context: AgentContext) -> AgentResult:
        documents = context.parsed_data or []
        features = context.extracted_features or {}
        query = context.raw_input or ""

        # ---- 如果 specialist agent 已产出结果，直接整合 ----
        if not documents and not features and context.intermediate_findings:
            documents = self._findings_to_documents(context.intermediate_findings, context.final_answer)
            if not documents:
                return AgentResult(
                    success=True,
                    message="直接使用 specialist 结果",
                    data={"markdown": context.final_answer or "", "report": {}, "sources": []},
                    context_updates={"final_answer": context.final_answer or query},
                )

        # ---- 防御性类型校正 ----
        if isinstance(documents, str):
            documents = [{"text": documents, "metadata": {"source": "direct_input"}}]
        if not isinstance(documents, list):
            documents = []
        documents = [d for d in documents if isinstance(d, dict)]

        if isinstance(features, str):
            features = {"raw": features}
        if not isinstance(features, dict):
            features = {}

        if not documents and not features:
            return AgentResult(success=False, message="无可用数据生成报告")

        # 1. 构建 source 列表
        sources = self._build_sources(documents)
        metrics = features.get("metrics", {})
        entities = features.get("entities", [])
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(entities, list):
            entities = []

        # 2. 委托工具 — LLM 报告生成
        try:
            report_result = self.call_tool(
                "synthesize_report",
                query=query,
                sources=sources,
                metrics=metrics,
                entities=entities,
            )
        except Exception as e:
            report_result = {"report": self._fallback_report_data(query, sources), "error": str(e)}

        report_json = report_result.get("report", {})
        check_result = report_result.get("hallucination_check", {})

        # 3. 渲染 Markdown
        markdown = self._render_markdown(report_json, sources)

        return AgentResult(
            success=True,
            message=f"报告生成完成 ({len(sources)} 个来源, {len(report_json.get('key_findings', []))} 个发现)",
            data={
                "report": report_json,
                "sources": sources,
                "markdown": markdown,
                "hallucination_check": check_result,
            },
            context_updates={
                "final_answer": markdown,
                "intermediate_findings": [
                    {"stage": "report", "source_count": len(sources),
                     "findings_count": len(report_json.get("key_findings", []))}
                ],
                "metadata": {
                    "report_source_count": len(sources),
                    "hallucination_risk": check_result.get("risk") if isinstance(check_result, dict) else None,
                },
            }
        )

    # ===================== 构建 source 列表 =====================

    def _findings_to_documents(self, findings: List[Dict], final_answer: Optional[str]) -> List[Dict]:
        """将 specialist agent 的 intermediate_findings 转为 source documents"""
        documents = []
        for f in findings:
            stage = f.get("stage", "unknown")
            if stage == "kline_analysis":
                ts_code = f.get("ts_code", "")
                stats = f.get("stats", {})
                indicators = f.get("indicators", {})
                text = (
                    f"K线技术分析: {ts_code}\n"
                    f"数据点: {f.get('data_points', 0)} 个交易日\n"
                    f"收盘价: {stats.get('latest_close', 'N/A')}\n"
                    f"区间涨跌: {stats.get('period_change_pct', 'N/A')}%\n"
                    f"MACD信号: {indicators.get('macd', {}).get('signal', 'N/A')}\n"
                    f"RSI: {indicators.get('rsi', {}).get('value', 'N/A')}"
                )
                documents.append({"text": text, "metadata": {"source": "KLineAgent", "stage": "kline"}})
            elif stage == "event_impact":
                event_count = f.get("event_count", 0)
                assessment = f.get("assessment", {})
                text = (
                    f"事件影响分析\n事件数: {event_count}\n"
                    f"综合判断: {assessment.get('overall_label', '未知')}\n"
                    f"综合影响力: {assessment.get('overall_factor', 0)}/10"
                )
                if assessment.get("summary"):
                    text += f"\n摘要: {assessment['summary']}"
                documents.append({"text": text, "metadata": {"source": "EventImpactAgent", "stage": "event_impact"}})
            elif stage == "coordination":
                pass  # 调度信息不生成报告 source
            else:
                text = f"分析结果 ({stage}): " + ", ".join(
                    f"{k}={v}" for k, v in f.items() if k != "stage"
                )
                documents.append({"text": text, "metadata": {"source": stage}})

        if final_answer and final_answer.strip():
            documents.append({
                "text": final_answer,
                "metadata": {"source": "specialist_summary", "title": "综合分析"},
            })
        return documents

    def _build_sources(self, documents: List[Dict]) -> List[Dict]:
        """将 parsed_data 转为带编号的 source 列表"""
        sources = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            meta = doc.get("metadata", doc.get("meta", {}))
            sources.append({
                "id": i,
                "text": text,
                "title": meta.get("title", text[:60]),
                "source": meta.get("source", "unknown"),
                "date": meta.get("date", meta.get("publish_time", "")),
                "keyword": meta.get("keyword", ""),
            })
        return sources

    # ===================== Fallback =====================

    def _fallback_report_data(self, query: str, sources: List[Dict]) -> Dict:
        """无工具时的基础报告数据"""
        findings = []
        for s in sources[:5]:
            findings.append({
                "finding": s["title"][:100],
                "importance": "medium",
                "source_refs": [s["id"]],
            })
        return {
            "title": query or "新闻分析报告",
            "key_findings": findings,
            "trend_analysis": f"共收集 {len(sources)} 条相关新闻",
            "sentiment": {"overall": "neutral", "reasoning": "需要 LLM"},
            "affected_sectors": [],
            "affected_companies": [],
            "contradictions": [],
            "summary": f"基于 {len(sources)} 条新闻源的基础分析。",
        }

    # ===================== Markdown 渲染 =====================

    def _render_markdown(self, report: Dict, sources: List[Dict]) -> str:
        """将报告 JSON 渲染为带引用的 Markdown"""
        lines = []

        title = report.get("title", "新闻分析报告")
        lines.append(f"# {title}\n")

        findings = report.get("key_findings", [])
        if findings:
            lines.append("## 关键发现\n")
            for f in findings:
                importance = f.get("importance", "medium")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance, "🟡")
                refs = f.get("source_refs", [])
                ref_str = " ".join(f"[{r}]" for r in refs) if refs else ""
                lines.append(f"- {icon} {f.get('finding', '')} {ref_str}")
            lines.append("")

        trend = report.get("trend_analysis", "")
        if trend:
            lines.append("## 趋势分析\n")
            lines.append(trend + "\n")

        sentiment = report.get("sentiment", {})
        if sentiment:
            overall = sentiment.get("overall", "neutral")
            sentiment_icon = {
                "positive": "📈 正面", "negative": "📉 负面",
                "neutral": "➡️ 中性", "mixed": "↕️ 混合"
            }.get(overall, "❓ 未知")
            lines.append("## 市场情绪\n")
            lines.append(f"**整体情绪: {sentiment_icon}**\n")
            reasoning = sentiment.get("reasoning", "")
            if reasoning:
                lines.append(reasoning + "\n")

        sectors = report.get("affected_sectors", [])
        companies = report.get("affected_companies", [])
        if sectors or companies:
            lines.append("## 影响范围\n")
            if sectors:
                lines.append(f"**行业:** {', '.join(sectors)}\n")
            if companies:
                lines.append(f"**公司:** {', '.join(companies)}\n")

        contradictions = report.get("contradictions", [])
        if contradictions:
            lines.append("## 矛盾信息\n")
            for c in contradictions:
                lines.append(f"- ⚠️ {c}")
            lines.append("")

        summary = report.get("summary", "")
        if summary:
            lines.append("## 总结\n")
            lines.append(summary + "\n")

        if sources:
            lines.append("---\n")
            lines.append("## 来源\n")
            for s in sources:
                date_str = f" ({s['date']})" if s.get("date") else ""
                source_str = f" — {s['source']}" if s.get("source") else ""
                lines.append(f"[{s['id']}] {s['title'][:80]}{date_str}{source_str}")

        return "\n".join(lines)
