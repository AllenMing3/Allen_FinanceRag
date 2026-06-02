"""
ReportAgent — 多格式报告生成

功能:
- 整合所有 Agent 结果
- 多格式输出: Markdown / JSON / HTML
- 带引用 + 置信度 + 防幻觉标注
"""
import json
import os
from typing import Dict, Any, List, Optional

from financial_rag.core.coordinator import BaseAgent, AgentContext, AgentResult
from financial_rag.core.reflector import HallucinationGuard


class ReportAgent(BaseAgent):
    """
    Agent 5: 报告生成

    输出格式:
    - Markdown: 人类可读，含图表描述
    - JSON: 结构化数据，供下游系统消费
    - HTML: 可视化展示（可选）
    """

    def __init__(self, model_router=None):
        super().__init__(
            name="ReportAgent",
            description="多格式分析报告生成"
        )
        self.model_router = model_router
        self.hallucination_guard = HallucinationGuard()

    def process(self, context: AgentContext) -> AgentResult:
        findings = context.intermediate_findings or []

        # 1. 整合所有阶段结果
        report_data = self._consolidate(context)

        # 2. 通过防幻觉校验
        if report_data.get("summary"):
            check = self.hallucination_guard.precheck(
                report_data["summary"],
                report_data.get("sources", [])
            )
            report_data["hallucination_check"] = check

        # 3. 生成多格式报告
        reports = {
            "markdown": self._render_markdown(report_data),
            "json": json.dumps(report_data, ensure_ascii=False, indent=2, default=str),
            "html": self._render_html(report_data),
        }

        return AgentResult(
            success=True,
            message="报告生成完成",
            data=reports,
            context_updates={
                "final_answer": reports["markdown"],
                "metadata": {
                    "report_formats": list(reports.keys()),
                    "hallucination_risk": report_data.get("hallucination_check", {}).get("warning", False),
                }
            }
        )

    def _consolidate(self, context: AgentContext) -> Dict:
        """整合所有中间发现为报告结构"""
        # 实际实现: 遍历 Agent 链的输出
        pass
        return {
            "title": "财务分析报告（待填充）",
            "summary": "",
            "sections": [],
            "sources": [],
            "confidence": 0.0,
        }

    def _render_markdown(self, data: Dict) -> str:
        """生成 Markdown 报告"""
        # 实际实现: 含标题层级、表格、指标卡片
        pass
        return ""

    def _render_html(self, data: Dict) -> str:
        """生成 HTML 报告"""
        # 实际实现: 带样式和交互
        pass
        return ""

    def save_report(self, path: str, fmt: str = "markdown"):
        """保存报告到文件"""
        # 实际实现
        pass

    def can_handle(self, context: AgentContext) -> bool:
        return len(context.intermediate_findings) >= 2
