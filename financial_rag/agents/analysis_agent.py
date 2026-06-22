"""
AnalysisAgent — 统一分析 Agent

合并 ExtractionAgent + KLineAgent + EventImpactAgent + ReportAgent 的全部能力。
Agent 只做编排决策，所有重活委托给 tools。

根据 context.metadata.intent (由 Coordinator 设置) 选择工具链:
- kline        → analyze_kline + generate_kline_analysis → synthesize_report
- event_impact → fetch_date_events + assess_event_impact → synthesize_report
- report/news  → extract_metrics + extract_entities + generate_queries → synthesize_report
- general      → extract_metrics + extract_entities → synthesize_report
"""
import re
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult
from financial_rag.tools.kline_tools import STOCK_MAP

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """
    统一分析 Agent — 信息抽取 + 专业分析 + 报告生成

    轻量级编排者:
    1. 读取 intent (Coordinator 注入)
    2. 按 intent 委托对应工具链
    3. 统一生成 Markdown 报告
    """

    # AI 行业指标体系 (用于质量评分)
    AI_METRICS = [
        "revenue", "net_income", "gross_margin", "rd_expense", "arr",
        "gpu_count", "training_cluster_size", "inference_cost_per_token",
        "model_params", "context_window", "api_calls", "customer_count",
    ]

    def __init__(self):
        super().__init__(
            name="AnalysisAgent",
            description="统一分析: 指标抽取 + K线分析 + 事件影响 + 报告生成",
        )

    def can_handle(self, context: AgentContext) -> bool:
        """始终可处理 — 由 Coordinator 通过 agent_chain 调度"""
        return True

    def process(self, context: AgentContext) -> AgentResult:
        """按 intent 分发到对应工具链，最终统一生成报告"""
        intent = context.metadata.get("intent", "general")
        query = context.raw_input or ""
        logger.info(f"[AnalysisAgent] 入口: intent={intent}, query={query[:80]!r}, "
                    f"parsed_docs={len(context.parsed_data or [])}, "
                    f"findings={len(context.intermediate_findings or [])}")

        # 按 intent 执行对应工具链
        if intent == "kline":
            findings = self._run_kline_chain(context)
        elif intent == "event_impact":
            findings = self._run_event_chain(context)
        elif intent == "news" and context.parsed_data:
            # 新闻解读: 已有加载文本时走深度分析
            return self._run_deep_news_chain(context)
        else:
            findings = self._run_extraction_chain(context)

        logger.info(f"[AnalysisAgent] 工具链完成: stage={findings.get('stage')}, "
                    f"has_error={'error' in findings}, keys={list(findings.keys())}")

        # 统一报告生成
        return self._generate_report(query, findings, context)

    # ===================== K 线工具链 =====================

    def _run_kline_chain(self, context: AgentContext) -> Dict:
        """K 线分析: analyze_kline → generate_kline_analysis"""
        ts_code = context.metadata.get("ts_code", "")
        name = context.metadata.get("stock_name", "")
        days = context.metadata.get("days", 60)

        if not ts_code:
            ts_code, name = self._extract_stock_code(context.raw_input)

        if not ts_code:
            return {"error": "无法识别股票代码", "stage": "kline_analysis"}

        # 获取 K 线数据 + 技术指标
        try:
            kline_data = self.call_tool("analyze_kline", ts_code=ts_code, days=days)
        except Exception as e:
            logger.warning(f"K线数据获取失败: {e}")
            return {"error": str(e), "stage": "kline_analysis"}

        if "error" in kline_data:
            return {"error": kline_data["error"], "stage": "kline_analysis"}

        stats = kline_data.get("stats", {})
        indicators = kline_data.get("indicators", {})

        # LLM 分析
        try:
            analysis_result = self.call_tool(
                "generate_kline_analysis",
                ts_code=ts_code, name=name or ts_code,
                stats=stats, indicators=indicators,
            )
            analysis = analysis_result.get("analysis", "")
        except Exception as e:
            logger.warning(f"K线分析生成失败: {e}")
            analysis = ""

        return {
            "stage": "kline_analysis",
            "ts_code": ts_code, "name": name or ts_code,
            "data_points": kline_data.get("data_points", 0),
            "stats": stats, "indicators": indicators,
            "analysis": analysis,
        }

    # ===================== 事件影响工具链 =====================

    def _run_event_chain(self, context: AgentContext) -> Dict:
        """事件影响: fetch_date_events → assess_event_impact"""
        date = context.metadata.get("date", "")
        stock_code = context.metadata.get("stock_code", "")
        stock_name = context.metadata.get("stock_name", "")
        keyword = context.metadata.get("keyword", "")

        if not date:
            date = self._extract_date(context.raw_input)
        if not stock_code:
            stock_code, stock_name = self._extract_stock(context.raw_input)

        if not date and not keyword:
            return {"error": "缺少日期或关键词", "stage": "event_impact"}

        # 获取事件
        try:
            events_data = self.call_tool("fetch_date_events", date=date, keyword=keyword)
        except Exception as e:
            logger.warning(f"事件获取失败: {e}")
            return {"error": str(e), "events": [], "stage": "event_impact"}

        events = events_data.get("events", [])
        if not events:
            return {"date": date, "events": [], "stage": "event_impact"}

        # 可选 K 线上下文
        kline_context = None
        if stock_code:
            try:
                kline_context = self.call_tool(
                    "fetch_kline_context", stock_code=stock_code, date=date, window_days=10,
                )
            except Exception as e:
                logger.warning(f"K线上下文获取失败: {e}")

        # 影响评估
        try:
            assessment = self.call_tool(
                "assess_event_impact", events=events,
                kline_context=kline_context, stock_name=stock_name,
            )
        except Exception as e:
            logger.warning(f"影响评估失败: {e}")
            assessment = {"overall_label": "未知", "overall_factor": 0}

        return {
            "stage": "event_impact",
            "date": date, "stock_code": stock_code, "stock_name": stock_name,
            "event_count": len(events), "events": events,
            "kline_context": kline_context, "assessment": assessment,
        }

    # ===================== 深度新闻分析工具链 =====================

    def _run_deep_news_chain(self, context: AgentContext) -> AgentResult:
        """新闻解读: 调用 analyze_news_deep 工具获取结构化多维分析"""
        query = context.raw_input or ""
        # 组合文本: parsed_data + intermediate_findings
        text_parts = []
        for d in (context.parsed_data or []):
            if isinstance(d, dict) and d.get("text"):
                text_parts.append(d["text"])
        for f in context.intermediate_findings:
            if isinstance(f, dict) and f.get("stage") != "extraction":
                text_parts.append(str({k: v for k, v in f.items() if k != "stage"}))
        combined = "\n\n".join(text_parts)

        if not combined:
            return AgentResult(
                success=False,
                message="无新闻文本可供分析",
                context_updates={"final_answer": "无新闻文本可供分析"},
            )

        try:
            result = self.call_tool("analyze_news_deep", text=combined, query=query)
        except Exception as e:
            logger.error(f"[AnalysisAgent] analyze_news_deep 失败: {e}")
            result = {"error": str(e), "structured": {}, "analysis": f"分析失败: {e}"}

        structured = result.get("structured", {})
        analysis_text = result.get("analysis", "")
        success = not result.get("error")

        markdown = self._render_structured_news(structured, analysis_text)

        return AgentResult(
            success=success,
            message=f"深度新闻分析完成: {structured.get('verdict', 'N/A')}",
            data={"structured": structured, "analysis": analysis_text, "markdown": markdown},
            context_updates={
                "final_answer": markdown,
                "intermediate_findings": [{"stage": "deep_news", "success": success}],
                "metadata": {
                    "analysis_mode": "deep_news",
                    "verdict": structured.get("verdict", "N/A"),
                    "confidence": structured.get("confidence", "N/A"),
                },
            },
        )

    # ===================== 深度话题调研工具链 =====================

    def _run_deep_topic_chain(self, context: AgentContext) -> AgentResult:
        """话题调研: 调用 analyze_topic_deep 工具获取结构化调研结果"""
        query = context.raw_input or ""
        topic = context.metadata.get("topic", "") or query
        max_news = context.metadata.get("max_news", 20)

        if not topic:
            return AgentResult(
                success=False,
                message="无话题关键词",
                context_updates={"final_answer": "无话题关键词"},
            )

        try:
            result = self.call_tool("analyze_topic_deep", topic=topic, max_news=max_news)
        except Exception as e:
            logger.error(f"[AnalysisAgent] analyze_topic_deep 失败: {e}")
            result = {"error": str(e), "structured": {}, "analysis": f"调研失败: {e}"}

        structured = result.get("structured", {})
        analysis_text = result.get("analysis", "")
        success = not result.get("error")

        markdown = self._render_structured_topic(structured, analysis_text)

        return AgentResult(
            success=success,
            message=f"深度话题调研完成: {structured.get('verdict', 'N/A')}",
            data={"structured": structured, "analysis": analysis_text, "markdown": markdown},
            context_updates={
                "final_answer": markdown,
                "intermediate_findings": [{"stage": "deep_topic", "success": success}],
                "metadata": {
                    "analysis_mode": "deep_topic",
                    "verdict": structured.get("verdict", "N/A"),
                    "confidence": structured.get("confidence", "N/A"),
                },
            },
        )

    # ===================== 抽取 + 报告工具链 =====================

    def _run_extraction_chain(self, context: AgentContext) -> Dict:
        """抽取: extract_metrics + extract_entities in parallel, then generate_queries"""
        documents = context.parsed_data or []
        if not documents:
            # 尝试从 intermediate_findings 获取
            documents = self._findings_to_documents(
                context.intermediate_findings, context.final_answer
            )

        combined_text = "\n\n".join(
            d.get("text", "") for d in documents if isinstance(d, dict) and d.get("text")
        )

        if not combined_text:
            return {"metrics": {}, "entities": {}, "queries": [],
                    "documents": documents, "stage": "extraction"}

        # Parallel: extract metrics + entities simultaneously
        metrics = {}
        entities = {}

        def _extract_metrics():
            try:
                return self.call_tool("extract_financial_metrics", text=combined_text)
            except Exception as e:
                logger.warning(f"指标抽取失败: {e}")
                return {}

        def _extract_entities():
            try:
                return self.call_tool("extract_entities", text=combined_text)
            except Exception as e:
                logger.warning(f"实体抽取失败: {e}")
                return {}

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_metrics = ex.submit(_extract_metrics)
            f_entities = ex.submit(_extract_entities)
            metrics = f_metrics.result()
            entities = f_entities.result()

        # Sequential: generate queries (needs metrics + entities)
        queries = []
        try:
            queries = self.call_tool(
                "generate_search_queries", text=combined_text,
                metrics=metrics, entities=entities,
            )
        except Exception as e:
            logger.warning(f"查询生成失败: {e}")

        # 质量评分
        extraction_score = self._evaluate_extraction(metrics, entities)
        query_score = self._evaluate_queries(queries)

        return {
            "stage": "extraction",
            "metrics": metrics, "entities": entities, "queries": queries,
            "documents": documents,
            "_scores": {"extraction": extraction_score, "query_rewrite": query_score},
        }

    # ===================== 统一报告生成 =====================

    def _generate_report(self, query: str, findings: Dict, context: AgentContext) -> AgentResult:
        """所有工具链的结果统一生成报告"""
        # 错误快速返回
        if findings.get("error") and not findings.get("stage"):
            return AgentResult(
                success=False, message=findings["error"],
                context_updates={"final_answer": findings["error"]},
            )

        stage = findings.get("stage", "unknown")

        # K 线 / 事件: 直接格式化
        if stage == "kline_analysis":
            answer = findings.get("analysis", "")
            return AgentResult(
                success=True,
                message=f"K线分析完成: {findings.get('name', '')} ({findings.get('data_points', 0)} 日)",
                data=findings,
                context_updates={
                    "final_answer": answer,
                    "intermediate_findings": [{**findings, "success": True}],
                },
            )

        if stage == "event_impact":
            answer = self._format_event_answer(findings)
            return AgentResult(
                success=bool(findings.get("events")),
                message=f"事件影响分析: {findings.get('event_count', 0)} 个事件",
                data=findings,
                context_updates={
                    "final_answer": answer,
                    "intermediate_findings": [{**findings, "success": True}],
                },
            )

        # 抽取 → 报告
        documents = findings.get("documents", [])
        metrics = findings.get("metrics", {})
        entities = findings.get("entities", {})

        if isinstance(documents, str):
            documents = [{"text": documents, "meta": {"source": "direct_input"}}]
        documents = [d for d in documents if isinstance(d, dict)]

        sources = self._build_sources(documents)

        # LLM 报告生成
        logger.info(f"[AnalysisAgent._generate_report] sources={len(sources)}, "
                    f"metrics_type={type(metrics).__name__}, entities_type={type(entities).__name__}")
        try:
            report_result = self.call_tool(
                "synthesize_report", query=query, sources=sources,
                metrics=metrics if isinstance(metrics, dict) else {},
                entities=entities if isinstance(entities, dict) else {},
            )
        except Exception as e:
            report_result = {"report": self._fallback_report(query, sources), "error": str(e)}

        report_json = report_result.get("report", {})
        check_result = report_result.get("hallucination_check", {})
        markdown = self._render_markdown(report_json, sources)

        return AgentResult(
            success=True,
            message=f"报告生成: {len(sources)} 来源, {len(report_json.get('key_findings', []))} 发现",
            data={
                "report": report_json, "sources": sources, "markdown": markdown,
                "hallucination_check": check_result,
                "_scores": findings.get("_scores", {}),
            },
            context_updates={
                "final_answer": markdown,
                "extracted_features": {"metrics": metrics, "entities": entities,
                                       "queries": findings.get("queries", [])},
                "intermediate_findings": [{
                    "stage": "extraction",
                    "success": True,
                    "source_count": len(sources),
                    "extraction_score": findings.get("_scores", {}).get("extraction", 0),
                }],
                "metadata": {
                    "report_source_count": len(sources),
                    "hallucination_risk": check_result.get("risk") if isinstance(check_result, dict) else None,
                },
            },
        )

    # ===================== 辅助方法 =====================

    def _build_sources(self, documents: List[Dict]) -> List[Dict]:
        sources = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            meta = doc.get("meta", {})
            sources.append({
                "id": i, "text": text,
                "title": meta.get("title", text[:60]),
                "source": meta.get("source", "unknown"),
                "date": meta.get("date", meta.get("publish_time", "")),
                "keyword": meta.get("keyword", ""),
            })
        return sources

    def _findings_to_documents(self, findings: List[Dict], final_answer: Optional[str]) -> List[Dict]:
        """将 intermediate_findings 转为 source documents"""
        documents = []
        for f in (findings or []):
            stage = f.get("stage", "unknown")
            if stage == "coordination":
                continue
            text = f"分析结果 ({stage}): " + ", ".join(
                f"{k}={v}" for k, v in f.items() if k != "stage"
            )
            documents.append({"text": text, "meta": {"source": stage}})
        if final_answer and final_answer.strip():
            documents.append({"text": final_answer, "meta": {"source": "specialist_summary"}})
        return documents

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
            label = {"positive": "📈 正面", "negative": "📉 负面",
                     "neutral": "➡️ 中性", "mixed": "↕️ 混合"}.get(overall, "❓ 未知")
            lines.append("## 市场情绪\n")
            lines.append(f"**整体情绪: {label}**\n")
            if sentiment.get("reasoning"):
                lines.append(sentiment["reasoning"] + "\n")

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

    def _render_structured_news(self, structured: Dict, analysis_text: str) -> str:
        """将深度新闻分析结构化数据渲染为 Markdown"""
        lines = []
        verdict = structured.get("verdict", "N/A")
        confidence = structured.get("confidence", "N/A")
        v_icon = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(verdict, "❓")
        lines.append(f"## 综合研判: {v_icon} {verdict} (置信度: {confidence})\n")

        # 多维影响
        impact = structured.get("impact", {})
        if impact:
            lines.append("### 多维影响\n")
            for dim in ["industry", "company", "tech", "market"]:
                d = impact.get(dim, {})
                if d:
                    dir_label = {"bullish": "📈 利好", "bearish": "📉 利空", "neutral": "➡️ 中性"}.get(d.get("direction", ""), "❓")
                    dim_cn = {"industry": "行业", "company": "公司", "tech": "技术", "market": "市场"}.get(dim, dim)
                    lines.append(f"- **{dim_cn}**: {dir_label} — {d.get('summary', '')}")
            lines.append("")

        # 关键信号
        signals = structured.get("key_signals", [])
        if signals:
            lines.append("### 关键信号\n")
            for s in signals:
                sev = s.get("severity", 3)
                bar = "■" * sev + "□" * (5 - sev)
                sig_type = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️"}.get(s.get("type", ""), "•")
                lines.append(f"- {sig_type} [{bar}] {s.get('signal', '')}")
            lines.append("")

        # 综合分析
        if analysis_text:
            lines.append("### 综合分析\n")
            lines.append(analysis_text + "\n")

        # 风险 + 后续关注
        risks = structured.get("risks", [])
        watch = structured.get("watch_next", [])
        if risks:
            lines.append("### 风险提示\n")
            for r in risks:
                lines.append(f"- ⚠️ {r}")
            lines.append("")
        if watch:
            lines.append("### 后续关注\n")
            for w in watch:
                lines.append(f"- 👀 {w}")
            lines.append("")

        return "\n".join(lines)

    def _render_structured_topic(self, structured: Dict, analysis_text: str) -> str:
        """将深度话题调研结构化数据渲染为 Markdown"""
        lines = []
        verdict = structured.get("verdict", "N/A")
        confidence = structured.get("confidence", "N/A")
        trend = structured.get("sentiment_trend", "unknown")
        trend_icon = {"improving": "📈", "deteriorating": "📉", "stable": "➡️", "mixed": "↕️"}.get(trend, "❓")
        lines.append(f"## 话题调研: {verdict} (置信度: {confidence}, 情绪趋势: {trend_icon} {trend})\n")

        # 子话题
        subs = structured.get("sub_topics", [])
        if subs:
            lines.append("### 子话题聚类\n")
            for s in subs:
                sent = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(s.get("sentiment", ""), "⚪")
                lines.append(f"- {sent} **{s.get('name', '')}**: {s.get('summary', '')}")
            lines.append("")

        # 关键参与者
        players = structured.get("key_players", [])
        if players:
            lines.append("### 关键参与者\n")
            lines.append("| 名称 | 角色 | 提及次数 |")
            lines.append("|------|------|---------|")
            for p in players:
                lines.append(f"| {p.get('name', '')} | {p.get('role', '')} | {p.get('mentions', 0)} |")
            lines.append("")

        # 综合分析
        if analysis_text:
            lines.append("### 综合分析\n")
            lines.append(analysis_text + "\n")

        # 投资启示
        implication = structured.get("investment_implication", "")
        if implication:
            lines.append("### 投资启示\n")
            lines.append(implication + "\n")

        # 反向信号
        contra = structured.get("contrarian_signals", [])
        if contra:
            lines.append("### 反向信号\n")
            for c in contra:
                lines.append(f"- 🔍 {c}")
            lines.append("")

        # 风险
        risks = structured.get("risks", [])
        if risks:
            lines.append("### 风险提示\n")
            for r in risks:
                lines.append(f"- ⚠️ {r}")
            lines.append("")

        return "\n".join(lines)

    def _format_event_answer(self, data: Dict) -> str:
        """事件影响分析格式化"""
        lines = []
        date = data.get("date", "未知日期")
        stock = data.get("stock_name", "") or data.get("stock_code", "")
        assessment = data.get("assessment", {})

        lines.append(f"## 事件影响分析: {date}")
        if stock:
            lines.append(f"标的: {stock}")
        lines.append("")

        events = data.get("events", [])
        lines.append(f"### 当日事件 ({len(events)} 条)")
        for i, e in enumerate(events[:5], 1):
            lines.append(f"{i}. {e.get('title', '')[:60]}")
        lines.append("")

        lines.append("### 综合判断")
        overall = assessment.get("overall_label", "未知")
        overall_factor = assessment.get("overall_factor", 0)
        lines.append(f"**{overall}** (综合影响力: {overall_factor}/10)")
        if assessment.get("summary"):
            lines.append(f"> {assessment['summary']}")
        lines.append("\n*以上分析仅供参考，不构成投资建议。*")

        return "\n".join(lines)

    def _fallback_report(self, query: str, sources: List[Dict]) -> Dict:
        findings = [{"finding": s["title"][:100], "importance": "medium",
                      "source_refs": [s["id"]]} for s in sources[:5]]
        return {
            "title": query or "新闻分析报告",
            "key_findings": findings,
            "trend_analysis": f"共收集 {len(sources)} 条相关新闻",
            "sentiment": {"overall": "neutral", "reasoning": "需要 LLM"},
            "summary": f"基于 {len(sources)} 条新闻源的基础分析。",
        }

    # ===================== 质量评分 =====================

    def _evaluate_extraction(self, metrics: Dict, entities: Dict) -> float:
        metric_hit = sum(1 for m in self.AI_METRICS if m in metrics)
        metric_rate = metric_hit / max(len(self.AI_METRICS), 1)
        entity_categories = ["companies", "persons", "ai_models", "chips_hardware",
                             "tech_terms", "financial_figures", "event"]
        entity_hit = sum(
            1 for cat in entity_categories
            if entities.get(cat) and str(entities.get(cat)) not in ("[]", "{}")
        )
        entity_rate = min(entity_hit / 4.0, 1.0)
        return 0.6 * metric_rate + 0.4 * entity_rate

    def _evaluate_queries(self, queries: List[str]) -> float:
        if not queries:
            return 0.0
        count_score = min(1.0, len(queries) / 3)
        lengths = [len(q) for q in queries]
        diversity = 1.0 if len(set(lengths)) > 1 else 0.6
        return 0.4 * count_score + 0.6 * diversity

    # ===================== 提取辅助 =====================

    def _extract_stock_code(self, query: str) -> tuple:
        code_match = re.search(r'(\d{6})', query)
        if code_match:
            code = code_match.group(1)
            ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
            return ts_code, code
        for keyword, (ts_code, name) in STOCK_MAP.items():
            if keyword in query:
                return ts_code, name
        return "", ""

    def _extract_date(self, query: str) -> str:
        for pat in [
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
            r'(\d{4})年(\d{1,2})月(\d{1,2})日?',
            r'(\d{4})(\d{2})(\d{2})',
        ]:
            m = re.search(pat, query)
            if m:
                y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
                return f"{y}-{mo}-{d}"
        return ""

    def _extract_stock(self, query: str) -> tuple:
        for keyword, (ts_code, name) in STOCK_MAP.items():
            if keyword in query:
                return ts_code, name
        m = re.search(r'(\d{6})', query)
        if m:
            code = m.group(1)
            return (f"{code}.SH", code) if code.startswith("6") else (f"{code}.SZ", code)
        return "", ""
