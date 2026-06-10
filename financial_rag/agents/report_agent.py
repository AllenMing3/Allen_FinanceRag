"""
ReportAgent — LLM 驱动的新闻综合分析 + 引用报告

功能:
- 将 parsed_data (KB 文档) + extracted_features (指标/实体) 综合为上下文
- 调用 LLM 生成结构化分析报告 (JSON)
- 渲染为带引用的 Markdown 报告
- HallucinationGuard 防幻觉校验
"""
import json
import re
from typing import Dict, Any, List, Optional

from financial_rag.config import config
from financial_rag.core.base import BaseAgent, AgentContext, AgentResult
from financial_rag.llm.dashscope_client import get_llm
from financial_rag.core.reflector import HallucinationGuard
from financial_rag.prompts import (
    NEWS_SYNTHESIS_SYSTEM,
    NEWS_SYNTHESIS_PROMPT,
)


class ReportAgent(BaseAgent):
    """
    Agent 3: 新闻综合分析 + 报告生成

    接收 IngestionAgent 的 parsed_data 和 ExtractionAgent 的 extracted_features,
    调用 LLM 生成结构化的新闻分析报告, 带引用标注。
    """

    def __init__(self):
        super().__init__(
            name="ReportAgent",
            description="LLM 驱动的新闻综合分析 + 引用报告"
        )
        self._llm = None
        self.hallucination_guard = HallucinationGuard()

    def _get_llm(self):
        """懒加载 LLM 实例"""
        if self._llm is None:
            try:
                self._llm = get_llm(
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                    temperature=0.1,
                )
            except (ImportError, ValueError):
                self._llm = None
        return self._llm

    def process(self, context: AgentContext) -> AgentResult:
        documents = context.parsed_data or []
        features = context.extracted_features or {}
        query = context.raw_input or ""

        # ---- 防御性类型校正 ----
        if isinstance(documents, str):
            documents = [{"text": documents, "metadata": {"source": "direct_input"}}]
        if not isinstance(documents, list):
            documents = []
        # 过滤掉非 dict 的项
        documents = [d for d in documents if isinstance(d, dict)]

        if isinstance(features, str):
            features = {"raw": features}
        if not isinstance(features, dict):
            features = {}

        if not documents and not features:
            return AgentResult(success=False, message="无可用数据生成报告")

        try:
            # 1. 整合上下文为 source 列表
            sources = self._build_sources(documents)
            metrics = features.get("metrics", {})
            entities = features.get("entities", [])
            if not isinstance(metrics, dict):
                metrics = {}
            if not isinstance(entities, list):
                entities = []

            # 2. 调用 LLM 生成分析报告
            llm = self._get_llm()
            if llm:
                report_json = self._generate_report(llm, query, sources, metrics, entities)
            else:
                report_json = self._fallback_report(query, sources, metrics, entities)

            # 3. 渲染 Markdown
            markdown = self._render_markdown(report_json, sources)

            # 4. HallucinationGuard 校验
            check_result = {}
            if isinstance(report_json, dict) and report_json.get("summary"):
                source_texts = [s.get("text", "")[:200] for s in sources if isinstance(s, dict)]
                check_result = self.hallucination_guard.precheck(
                    report_json["summary"],
                    source_texts,
                )
        except Exception as e:
            print(f"[ReportAgent] 报告生成异常，走 fallback: {e}")
            sources = self._build_sources(documents)
            report_json = self._fallback_report(query, sources, {}, [])
            markdown = self._render_markdown(report_json, sources)
            check_result = {}

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
                    "hallucination_risk": check_result.get("warning", False),
                },
            }
        )

    # ===================== 构建 source 列表 =====================

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

    # ===================== LLM 报告生成 =====================

    def _generate_report(self, llm, query: str, sources: List[Dict],
                         metrics: Dict, entities: List[Dict]) -> Dict:
        """调用 LLM 生成结构化分析报告"""

        # 格式化 sources 为编号列表
        source_block = self._format_sources_for_prompt(sources)

        # 格式化 metrics
        metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2) if metrics else "无"

        # 格式化 entities
        entities_str = json.dumps(
            [{"type": e.get("type", ""), "data": e.get("data", {})} for e in entities[:10]],
            ensure_ascii=False, indent=2,
        ) if entities else "无"

        prompt = NEWS_SYNTHESIS_PROMPT.format(
            query=query or "综合新闻分析",
            sources=source_block,
            metrics=metrics_str,
            entities=entities_str,
        )

        try:
            response = llm.chat(
                messages=prompt,
                system=NEWS_SYNTHESIS_SYSTEM,
                max_tokens=2048,
                temperature=0.1,
            )
            content = response.content.strip()

            # 提取 JSON 对象
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[ReportAgent] LLM 报告生成失败: {e}")

        # LLM 失败时返回 fallback
        return self._fallback_report(query, sources, metrics, entities)

    def _format_sources_for_prompt(self, sources: List[Dict]) -> str:
        """格式化 sources 为 LLM prompt 中的编号文本"""
        parts = []
        for s in sources:
            header = f"[{s['id']}] {s['title']}"
            if s.get("date"):
                header += f" ({s['date']})"
            if s.get("source"):
                header += f" — 来源: {s['source']}"
            # 截取前 800 字符保留关键信息
            text = s["text"][:800]
            parts.append(f"{header}\n{text}")

        result = "\n\n".join(parts)
        # 限制总长度避免超 token
        if len(result) > 12000:
            result = result[:12000] + "\n... (更多来源已截断)"
        return result

    # ===================== Fallback (无 LLM) =====================

    def _fallback_report(self, query: str, sources: List[Dict],
                         metrics: Dict, entities: List[Dict]) -> Dict:
        """无 LLM 时生成基础报告"""
        findings = []
        for s in sources[:5]:
            findings.append({
                "finding": s["title"][:100],
                "importance": "medium",
                "source_refs": [s["id"]],
            })

        companies = []
        for e in entities:
            if e.get("type") == "company":
                name = e.get("data", {}).get("name", "")
                if name:
                    companies.append(name)

        return {
            "title": query or "新闻分析报告",
            "key_findings": findings,
            "trend_analysis": f"共收集 {len(sources)} 条相关新闻 (无 LLM，趋势分析不可用)",
            "sentiment": {"overall": "neutral", "reasoning": "需要 LLM 进行情绪分析"},
            "affected_sectors": [],
            "affected_companies": companies[:5],
            "contradictions": [],
            "summary": f"基于 {len(sources)} 条新闻源的基础分析。启用 LLM 可获得更详细的趋势和情绪分析。",
        }

    # ===================== Markdown 渲染 =====================

    def _render_markdown(self, report: Dict, sources: List[Dict]) -> str:
        """将报告 JSON 渲染为带引用的 Markdown"""
        lines = []

        # 标题
        title = report.get("title", "新闻分析报告")
        lines.append(f"# {title}\n")

        # 关键发现
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

        # 趋势分析
        trend = report.get("trend_analysis", "")
        if trend:
            lines.append("## 趋势分析\n")
            lines.append(trend + "\n")

        # 市场情绪
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

        # 影响范围
        sectors = report.get("affected_sectors", [])
        companies = report.get("affected_companies", [])
        if sectors or companies:
            lines.append("## 影响范围\n")
            if sectors:
                lines.append(f"**行业:** {', '.join(sectors)}\n")
            if companies:
                lines.append(f"**公司:** {', '.join(companies)}\n")

        # 矛盾信息
        contradictions = report.get("contradictions", [])
        if contradictions:
            lines.append("## 矛盾信息\n")
            for c in contradictions:
                lines.append(f"- ⚠️ {c}")
            lines.append("")

        # 总结
        summary = report.get("summary", "")
        if summary:
            lines.append("## 总结\n")
            lines.append(summary + "\n")

        # 来源引用
        if sources:
            lines.append("---\n")
            lines.append("## 来源\n")
            for s in sources:
                date_str = f" ({s['date']})" if s.get("date") else ""
                source_str = f" — {s['source']}" if s.get("source") else ""
                lines.append(f"[{s['id']}] {s['title'][:80]}{date_str}{source_str}")

        return "\n".join(lines)

    def can_handle(self, context: AgentContext) -> bool:
        """需要至少 parsed_data 或 extracted_features"""
        return bool(context.parsed_data or context.extracted_features)
