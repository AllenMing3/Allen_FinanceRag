"""ScoringAgent — 通用公共评分 Agent

职责:
- 对 Pipeline 各阶段结果进行质量评分
- 对最终输出进行防幻觉校验
- 生成可读的评分报告

设计原则:
- 公共能力，任何 feature 都能接入
- 只做 3 个 tool call，不写实现逻辑
- 所有预加工由上游通过 context.metadata 传入

通用接口字段 (通过 metadata 传入):
- scoring_source_items: 防幻觉 grounding 源
- scoring_mode: Guard 模式 ("rag" / "analysis")
- scoring_text: 待校验的完整文本

RAG 场景兼容: 如果没有通用字段，自动 fallback 到 retrieved_items / final_answer。

Agent 只做编排决策，所有评分逻辑委托给 tools:
- evaluate_pipeline_quality: 各阶段打分
- check_hallucination: 防幻觉校验
- generate_score_report: 评分报告生成
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult

logger = logging.getLogger(__name__)


class ScoringAgent(BaseAgent):
    """
    全链路评分 Agent

    轻量级编排者:
    1. 从 context 获取各阶段结果
    2. 调用 evaluate_pipeline_quality 打分
    3. 调用 check_hallucination 防幻觉
    4. 调用 generate_score_report 生成报告
    """

    def __init__(self):
        super().__init__(
            name="ScoringAgent",
            description="全链路评分: 各阶段质量打分 + 防幻觉校验 + 评分报告",
        )

    def can_handle(self, context: AgentContext) -> bool:
        """有 final_answer 或中间结果时进行评分"""
        return bool(context.final_answer or context.intermediate_findings)

    def process(self, context: AgentContext) -> AgentResult:
        """执行全链路评分 — 全部委托给工具

        读取通用接口字段（优先）或 RAG 场景字段（fallback），
        然后调用 3 个评分工具。
        """
        metadata = context.metadata

        # --- 读取通用接口字段（任何 feature 都能传入）---
        source_items = metadata.get("scoring_source_items")
        guard_mode = metadata.get("scoring_mode", "rag")
        check_text = metadata.get("scoring_text") or context.final_answer or ""

        # --- RAG 场景 fallback ---
        fetched_data = metadata.get("fetched_data", [])
        retrieved_items = metadata.get("retrieved_items", [])
        fill_stats = metadata.get("fill_stats")

        if not source_items:
            source_items = [
                {"text": it["text"]}
                for it in retrieved_items
                if isinstance(it, dict) and it.get("text")
            ]

        # 从 agent 结果中提取成功/失败状态
        agent_results = [
            {"success": f.get("success", True), "agent_name": f.get("stage", "unknown")}
            for f in context.intermediate_findings
        ]

        # Parallel: evaluate_pipeline_quality ‖ check_hallucination (no data dependency)
        def _evaluate():
            return self.call_tool(
                "evaluate_pipeline_quality",
                fetched_data=fetched_data,
                retrieved_items=retrieved_items,
                agent_results=agent_results if agent_results else None,
                fill_stats=fill_stats,
                fetch_elapsed_ms=metadata.get("fetch_elapsed_ms", 0),
                index_elapsed_ms=metadata.get("index_elapsed_ms", 0),
                process_elapsed_ms=metadata.get("process_elapsed_ms", 0),
                output_elapsed_ms=metadata.get("output_elapsed_ms", 0),
            )

        def _check_hallucination():
            if not check_text:
                return {}
            return self.call_tool(
                "check_hallucination",
                output_text=check_text,
                source_items=source_items if source_items else None,
                mode=guard_mode,
            )

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_eval = ex.submit(_evaluate)
            f_hall = ex.submit(_check_hallucination)
            pipeline_scores = f_eval.result()
            hallucination_check = f_hall.result()

        # Sequential: generate_score_report (depends on both parallel results)
        report_result = self.call_tool(
            "generate_score_report",
            pipeline_scores=pipeline_scores,
            hallucination_check=hallucination_check if hallucination_check else None,
            query=context.raw_input or "",
        )

        report = report_result.get("report", "评分报告生成失败")

        # Build user-visible hallucination report from guard
        hallucination_report = ""
        if isinstance(hallucination_check, dict):
            hallucination_report = hallucination_check.get("report", "")

        # Derive success from execution (did scoring complete?) not from scores themselves
        grade = pipeline_scores.get("grade", "N/A") if isinstance(pipeline_scores, dict) else "N/A"
        hallucination_risk = hallucination_check.get("risk", "unknown") if isinstance(hallucination_check, dict) else "unknown"
        any_agent_failed = any(not ar.get("success", True) for ar in agent_results)
        # Success = scoring ran to completion. Quality info lives in data.
        scoring_success = bool(pipeline_scores) and grade != "N/A"

        return AgentResult(
            success=scoring_success,
            message=f"全链路评分完成 ({pipeline_scores.get('total_stages', 0)} 阶段)",
            data={
                "pipeline_scores": pipeline_scores,
                "hallucination_check": hallucination_check,
                "report": report,
                "hallucination_report": hallucination_report,
                "any_agent_failed": any_agent_failed,
            },
            context_updates={
                "metadata": {
                    "scoring_report": report,
                    "hallucination_risk": hallucination_check.get("risk", "unknown") if isinstance(hallucination_check, dict) else "unknown",
                    "hallucination_report": hallucination_report,
                    "pipeline_grade": pipeline_scores.get("grade", "N/A") if isinstance(pipeline_scores, dict) else "N/A",
                },
            },
        )
