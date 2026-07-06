"""
评分工具 — 供 ScoringAgent 调用的评分与防幻觉能力

工具:
- evaluate_pipeline_quality: 对 Pipeline 各阶段结果进行全链路打分
- check_hallucination: 对最终输出进行防幻觉校验
- generate_score_report: 生成可读的评分报告
"""
import logging
from typing import Dict, Any, List, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)

# LLM 注入引用，供 HallucinationGuard L5+L6 使用
_scoring_llm_ref = {"llm": None}


def inject_scoring_llm(llm):
    """注入 LLM 实例供防幻觉 L5/L6 层使用"""
    _scoring_llm_ref["llm"] = llm


# ===================== 工具实现 =====================


def evaluate_pipeline_quality(
    fetched_data: List = None,
    retrieved_items: List = None,
    agent_results: List = None,
    fill_stats: Dict = None,
    fetch_elapsed_ms: float = 0,
    index_elapsed_ms: float = 0,
    process_elapsed_ms: float = 0,
    output_elapsed_ms: float = 0,
) -> Dict:
    """对 Pipeline 各阶段进行全链路打分。

    各阶段评分规则:
    - fetch: 是否成功获取数据
    - index: 检索结果质量
    - process: Agent 分析成功率
    - output: 槽位填充率

    Args:
        fetched_data: Phase 1 获取的原始数据列表
        retrieved_items: Phase 2 检索返回的结果列表
        agent_results: Phase 3 各 Agent 执行结果列表 [{"success": bool, "agent_name": str}, ...]
        fill_stats: Phase 4 槽位填充统计 {"filled_slots": int, "total_slots": int, "template_name": str}
        fetch_elapsed_ms: Phase 1 耗时(ms)
        index_elapsed_ms: Phase 2 耗时(ms)
        process_elapsed_ms: Phase 3 耗时(ms)
        output_elapsed_ms: Phase 4 耗时(ms)
    """
    from financial_rag.core.scorer import PipelineScoreCard

    card = PipelineScoreCard(query="pipeline_evaluation")
    stages = []

    # Phase 1: 数据获取
    if fetched_data is not None:
        score = 0.9 if len(fetched_data) > 0 else 0.3
        card.record("fetch", "数据获取", score, elapsed_ms=fetch_elapsed_ms,
                     details={"item_count": len(fetched_data)})
        stages.append({"stage": "fetch", "score": score, "elapsed_ms": fetch_elapsed_ms})

    # Phase 2: 检索
    if retrieved_items is not None:
        score = 0.85 if len(retrieved_items) > 0 else 0.3
        card.record("index", "RAG检索", score, elapsed_ms=index_elapsed_ms,
                     details={"hit_count": len(retrieved_items)})
        stages.append({"stage": "index", "score": score, "elapsed_ms": index_elapsed_ms})

    # Phase 3: 加工
    if agent_results is not None:
        ok_count = sum(1 for r in agent_results if r.get("success", False))
        total = len(agent_results)
        score = 0.8 if total > 0 and ok_count > 0 else 0.0
        card.record("process", "Multi-Agent加工", score, elapsed_ms=process_elapsed_ms,
                     details={"agents_ok": f"{ok_count}/{total}"})
        stages.append({"stage": "process", "score": score, "elapsed_ms": process_elapsed_ms})

    # Phase 4: 输出
    if fill_stats is not None:
        filled = fill_stats.get("filled_slots", 0)
        total = max(fill_stats.get("total_slots", 1), 1)
        score = filled / total
        card.record("output", "槽位输出", score, elapsed_ms=output_elapsed_ms,
                     details={"template": fill_stats.get("template_name", "unknown")})
        stages.append({"stage": "output", "score": score, "elapsed_ms": output_elapsed_ms})

    # Compute grade from overall score
    from financial_rag.core.scorer import ScoreGrade
    overall = card.overall_score()
    grade = ScoreGrade.from_score(overall).value

    return {
        "stages": stages,
        "summary": card.summary(),
        "grade": grade,
        "total_stages": len(stages),
    }


def check_hallucination(output_text: str, source_items: List = None) -> Dict:
    """对最终输出进行防幻觉校验。

    检查维度:
    - 数值引用是否有来源支撑
    - 关键声明是否与源数据一致
    - 是否存在无源数据的具体数字

    Args:
        output_text: 最终生成的文本
        source_items: 源数据列表 [{"text": "...", ...}, ...]
    """
    from financial_rag.guard.reflector import HallucinationGuard

    llm = _scoring_llm_ref["llm"]
    guard = HallucinationGuard(llm=llm)
    sources = source_items or []
    result = guard.check(output_text, sources)

    return {
        "overall_score": result.get("overall_score", 1.0),
        "risk": result.get("risk", "low"),
        "checks": result.get("checks", {}),
        "report": result.get("report", ""),
        "unverified": result.get("unverified", []),
    }


def generate_score_report(
    pipeline_scores: Dict = None,
    hallucination_check: Dict = None,
    query: str = "",
) -> Dict:
    """生成可读的全链路评分报告。

    Args:
        pipeline_scores: evaluate_pipeline_quality 的返回结果
        hallucination_check: check_hallucination 的返回结果
        query: 原始查询
    """
    lines = []
    lines.append("## 全链路评分报告")
    if query:
        lines.append(f"查询: {query}")
    lines.append("")

    # Pipeline 各阶段评分
    if pipeline_scores:
        stages = pipeline_scores.get("stages", [])
        lines.append("### 各阶段评分")
        for s in stages:
            bar = "█" * int(s["score"] * 10) + "░" * (10 - int(s["score"] * 10))
            lines.append(f"- {s['stage']}: {bar} {s['score']:.0%} ({s['elapsed_ms']:.0f}ms)")
        lines.append("")

    # 防幻觉校验 — 直接使用 guard 生成的格式报告
    if hallucination_check:
        guard_report = hallucination_check.get("report", "")
        if guard_report:
            lines.append(guard_report)
        else:
            # fallback: 旧格式
            lines.append("\n### 防幻觉校验")
            risk = hallucination_check.get("risk", "unknown")
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "❓")
            lines.append(f"风险等级: {risk_emoji} {risk}")
            score = hallucination_check.get("overall_score", 1.0)
            lines.append(f"可信度: {score:.0%}")

    report_text = "\n".join(lines)
    return {"report": report_text, "pipeline_scores": pipeline_scores, "hallucination_check": hallucination_check}


# ===================== FunctionDef 定义 =====================

EVALUATE_PIPELINE_TOOL = FunctionDef(
    name="evaluate_pipeline_quality",
    description="对 Pipeline 各阶段（获取/检索/加工/输出）进行全链路打分，返回各阶段评分和汇总。",
    parameters={
        "type": "object",
        "properties": {
            "fetched_data": {"type": "array", "description": "Phase 1 获取的原始数据列表"},
            "retrieved_items": {"type": "array", "description": "Phase 2 检索结果列表"},
            "agent_results": {"type": "array", "description": "Phase 3 Agent 结果列表"},
            "fill_stats": {"type": "object", "description": "Phase 4 槽位填充统计"},
            "fetch_elapsed_ms": {"type": "number", "description": "Phase 1 耗时(ms)", "default": 0},
            "index_elapsed_ms": {"type": "number", "description": "Phase 2 耗时(ms)", "default": 0},
            "process_elapsed_ms": {"type": "number", "description": "Phase 3 耗时(ms)", "default": 0},
            "output_elapsed_ms": {"type": "number", "description": "Phase 4 耗时(ms)", "default": 0},
        },
        "required": [],
    },
    callback=evaluate_pipeline_quality,
    category="analysis",
    tags=["评分", "Pipeline", "质量"],
)

CHECK_HALLUCINATION_TOOL = FunctionDef(
    name="check_hallucination",
    description="对最终输出进行防幻觉校验，检查数值引用是否有来源支撑。",
    parameters={
        "type": "object",
        "properties": {
            "output_text": {"type": "string", "description": "最终生成的文本"},
            "source_items": {"type": "array", "description": "源数据列表"},
        },
        "required": ["output_text"],
    },
    callback=check_hallucination,
    category="analysis",
    tags=["评分", "防幻觉", "校验"],
)

GENERATE_SCORE_REPORT_TOOL = FunctionDef(
    name="generate_score_report",
    description="生成可读的全链路评分报告，汇总各阶段评分和防幻觉校验结果。",
    parameters={
        "type": "object",
        "properties": {
            "pipeline_scores": {"type": "object", "description": "evaluate_pipeline_quality 的返回"},
            "hallucination_check": {"type": "object", "description": "check_hallucination 的返回"},
            "query": {"type": "string", "description": "原始查询", "default": ""},
        },
        "required": [],
    },
    callback=generate_score_report,
    category="analysis",
    tags=["评分", "报告", "汇总"],
)

SCORING_TOOLS = [EVALUATE_PIPELINE_TOOL, CHECK_HALLUCINATION_TOOL, GENERATE_SCORE_REPORT_TOOL]
