"""
报告工具 — 供 ReportAgent 调用的 LLM 报告生成能力

提供:
- synthesize_report: LLM 驱动的新闻综合分析报告生成
"""
import json
import logging
from typing import Dict, Any, List, Optional

from financial_rag.tools.core import FunctionDef
from financial_rag.llm.caller import LLMCaller

logger = logging.getLogger(__name__)


# LLM 注入引用
_report_llm_ref = {"llm": None}


def inject_report_llm(llm):
    """注入 LLM 实例供报告工具使用"""
    _report_llm_ref["llm"] = llm


def synthesize_report(
    query: str = "",
    sources: List = None,
    metrics: Dict = None,
    entities: List = None,
) -> Dict:
    """使用 LLM 综合分析源数据，生成结构化新闻报告。

    无 LLM 时返回纯数据基础报告。

    Args:
        query: 用户查询
        sources: 源数据列表 [{"id": 1, "text": "...", "title": "...", "source": "...", "date": "..."}, ...]
        metrics: 抽取的财务指标
        entities: 抽取的实体列表
    """
    sources = sources or []
    metrics = metrics or {}
    entities = entities or []

    llm = _report_llm_ref["llm"]

    if not sources and not metrics:
        return {"error": "无可用数据生成报告", "report": {}, "method": "none"}

    if llm is None:
        return _fallback_report(query, sources, metrics, entities)

    from financial_rag.prompts import NEWS_SYNTHESIS_SYSTEM, NEWS_SYNTHESIS_PROMPT

    # 格式化 sources
    source_block = _format_sources(sources)
    metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2) if metrics else "无"
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
        caller = LLMCaller(llm)
        report_json = caller.call_json(
            prompt,
            system=NEWS_SYNTHESIS_SYSTEM,
            max_tokens=2048,
            temperature=0.1,
        )
        if not report_json:
            # JSON 解析失败，用纯文本内容兑底
            response = caller.call(
                prompt,
                system=NEWS_SYNTHESIS_SYSTEM,
                max_tokens=2048,
                temperature=0.1,
            )
            report_json = {"summary": response.content.strip(), "key_findings": [], "title": query}

        # HallucinationGuard 预检
        check_result = {}
        if report_json.get("summary"):
            from financial_rag.core.reflector import HallucinationGuard
            guard = HallucinationGuard()
            source_texts = [s.get("text", "")[:200] for s in sources if isinstance(s, dict)]
            check_result = guard.precheck(report_json["summary"], source_texts)

        return {
            "report": report_json,
            "hallucination_check": check_result,
            "method": "llm",
        }
    except Exception as e:
        logger.warning(f"LLM 报告生成失败: {e}")
        return _fallback_report(query, sources, metrics, entities)


def _format_sources(sources: List[Dict]) -> str:
    """格式化 sources 为 LLM prompt 文本"""
    parts = []
    for s in sources:
        header = f"[{s.get('id', '?')}] {s.get('title', '')}"
        if s.get("date"):
            header += f" ({s['date']})"
        if s.get("source"):
            header += f" — 来源: {s['source']}"
        text = s.get("text", "")[:800]
        parts.append(f"{header}\n{text}")
    result = "\n\n".join(parts)
    if len(result) > 12000:
        result = result[:12000] + "\n... (更多来源已截断)"
    return result


def _fallback_report(query, sources, metrics, entities) -> Dict:
    """无 LLM 时的基础报告"""
    findings = []
    for s in sources[:5]:
        findings.append({
            "finding": s.get("title", "")[:100],
            "importance": "medium",
            "source_refs": [s.get("id", 0)],
        })

    companies = []
    for e in entities:
        if isinstance(e, dict) and e.get("type") == "company":
            name = e.get("data", {}).get("name", "")
            if name:
                companies.append(name)

    return {
        "report": {
            "title": query or "新闻分析报告",
            "key_findings": findings,
            "trend_analysis": f"共收集 {len(sources)} 条相关新闻 (无 LLM)",
            "sentiment": {"overall": "neutral", "reasoning": "需要 LLM"},
            "affected_sectors": [],
            "affected_companies": companies[:5],
            "contradictions": [],
            "summary": f"基于 {len(sources)} 条新闻源的基础分析。",
        },
        "hallucination_check": {},
        "method": "fallback",
    }


SYNTHESIZE_REPORT_TOOL = FunctionDef(
    name="synthesize_report",
    description="使用 LLM 综合分析源数据，生成结构化新闻报告（含关键发现、趋势分析、情绪判断）。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户查询", "default": ""},
            "sources": {"type": "array", "description": "源数据列表"},
            "metrics": {"type": "object", "description": "抽取的财务指标"},
            "entities": {"type": "array", "description": "抽取的实体列表"},
        },
        "required": [],
    },
    callback=synthesize_report,
    category="analysis",
    tags=["报告", "LLM", "综合分析", "新闻"],
)

REPORT_TOOLS = [SYNTHESIZE_REPORT_TOOL]
