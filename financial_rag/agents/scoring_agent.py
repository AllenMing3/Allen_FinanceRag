"""
ScoringAgent — 全链路评分 Agent

职责:
- 对 Pipeline 各阶段结果进行质量评分
- 对最终输出进行防幻觉校验
- 生成可读的评分报告

Agent 只做编排决策，所有评分逻辑委托给 tools:
- evaluate_pipeline_quality: 各阶段打分
- check_hallucination: 防幻觉校验
- generate_score_report: 评分报告生成
"""
import logging
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
        """执行全链路评分 — 全部委托给工具"""
        metadata = context.metadata

        # 提取各阶段数据
        fetched_data = metadata.get("fetched_data", [])
        retrieved_items = metadata.get("retrieved_items", [])
        fill_stats = metadata.get("fill_stats")

        # 从 agent 结果中提取成功/失败状态
        agent_results = []
        for finding in context.intermediate_findings:
            agent_results.append({
                "success": True,
                "agent_name": finding.get("stage", "unknown"),
            })

        # Step 1: 委托工具 — Pipeline 各阶段打分
        pipeline_scores = self.call_tool(
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

        # Step 2: 委托工具 — 防幻觉校验
        hallucination_check = {}
        if context.final_answer:
            source_items = []
            for item in retrieved_items:
                if isinstance(item, dict) and item.get("text"):
                    source_items.append({"text": item["text"]})
            hallucination_check = self.call_tool(
                "check_hallucination",
                output_text=context.final_answer,
                source_items=source_items if source_items else None,
            )

        # Step 3: 委托工具 — 生成评分报告
        report_result = self.call_tool(
            "generate_score_report",
            pipeline_scores=pipeline_scores,
            hallucination_check=hallucination_check if hallucination_check else None,
            query=context.raw_input or "",
        )

        report = report_result.get("report", "评分报告生成失败")

        return AgentResult(
            success=True,
            message=f"全链路评分完成 ({pipeline_scores.get('total_stages', 0)} 阶段)",
            data={
                "pipeline_scores": pipeline_scores,
                "hallucination_check": hallucination_check,
                "report": report,
            },
            context_updates={
                "metadata": {
                    "scoring_report": report,
                    "hallucination_risk": hallucination_check.get("risk", "unknown"),
                    "pipeline_grade": pipeline_scores.get("grade", "N/A"),
                },
            },
        )
