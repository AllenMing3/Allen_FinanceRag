"""
报告生成 Agent - 负责整合所有结果生成最终报告
"""
from typing import List, Dict, Any
from datetime import datetime
from .base_agent import BaseAgent, AgentContext, AgentResponse


class ReportAgent(BaseAgent):
    """
    报告生成 Agent
    
    职责：
    1. 整合所有 Agent 的结果
    2. 生成结构化报告
    3. 支持多种输出格式
    4. 提供可视化建议
    """
    
    def __init__(self):
        super().__init__(
            name="ReportAgent",
            description="整合所有分析结果，生成结构化报告"
        )
        self.report_data: Dict = {}
        self.report_formats = ["markdown", "json", "html"]
    
    def process(self, context: AgentContext) -> AgentResponse:
        """
        生成报告
        
        整合 context 中的所有信息生成最终报告
        """
        # 收集所有信息
        self.report_data = self._collect_data(context)
        
        # 生成不同格式的报告
        reports = {}
        for fmt in self.report_formats:
            reports[fmt] = self._generate_report(fmt)
        
        # 更新上下文
        context.final_answer = reports.get("markdown", "")
        context.metadata['reports'] = reports
        
        return AgentResponse(
            success=True,
            data={
                "reports": reports,
                "summary": self._generate_summary(),
                "key_findings": self._extract_key_findings(),
                "next_steps": self._extract_next_steps(context)
            },
            message="报告生成完成",
            context_updates={
                "final_report": reports,
                "final_answer": reports.get("markdown", "")
            }
        )
    
    def _collect_data(self, context: AgentContext) -> Dict:
        """收集所有数据"""
        return {
            "timestamp": datetime.now().isoformat(),
            "raw_input": context.raw_input,
            "parsed_data_summary": self._summarize_parsed_data(context),
            "analysis_results": context.analysis_results,
            "solutions": context.metadata.get('solutions', []),
            "action_plan": context.metadata.get('action_plan', {}),
            "intermediate_findings": context.intermediate_findings
        }
    
    def _summarize_parsed_data(self, context: AgentContext) -> Dict:
        """汇总解析数据"""
        if not context.parsed_data:
            return {}
        
        # 如果是日志条目列表
        if isinstance(context.parsed_data, list):
            return {
                "total_entries": len(context.parsed_data),
                "type": "log_entries"
            }
        
        return {"type": "unknown", "data": str(context.parsed_data)[:200]}
    
    def _generate_report(self, format_type: str) -> str:
        """生成特定格式的报告"""
        if format_type == "markdown":
            return self._generate_markdown_report()
        elif format_type == "json":
            return self._generate_json_report()
        elif format_type == "html":
            return self._generate_html_report()
        else:
            return "不支持的格式"
    
    def _generate_markdown_report(self) -> str:
        """生成 Markdown 报告"""
        data = self.report_data
        
        report = f"""# 日志分析报告

> 生成时间: {data['timestamp']}

---

## 执行摘要

{self._generate_summary()}

---

## 详细发现

"""
        
        # 添加中间发现
        if data.get('intermediate_findings'):
            report += "### 关键发现\n\n"
            for i, finding in enumerate(data['intermediate_findings'], 1):
                report += f"{i}. {finding}\n"
            report += "\n"
        
        # 添加分析结果
        if data.get('analysis_results'):
            report += "### 分析结果\n\n"
            analyses = data['analysis_results'].get('analyses', [])
            for analysis in analyses:
                report += self._format_analysis_section(analysis)
        
        # 添加解决方案
        if data.get('solutions'):
            report += "## 解决方案\n\n"
            solutions = data['solutions']
            
            # 高优先级
            high_priority = [s for s in solutions if s.get('priority') == 'high']
            if high_priority:
                report += "### 🔴 高优先级\n\n"
                for sol in high_priority:
                    report += self._format_solution(sol)
            
            # 中优先级
            medium_priority = [s for s in solutions if s.get('priority') == 'medium']
            if medium_priority:
                report += "### 🟡 中优先级\n\n"
                for sol in medium_priority:
                    report += self._format_solution(sol)
        
        # 添加行动计划
        if data.get('action_plan'):
            report += "## 行动计划\n\n"
            plan = data['action_plan']
            report += f"**预估总时间**: {plan.get('estimated_total_time', '未知')}\n\n"
            
            if plan.get('recommended_sequence'):
                report += "### 推荐执行顺序\n\n"
                for item in plan['recommended_sequence']:
                    report += f"- [ ] {item}\n"
                report += "\n"
        
        # 添加结论
        report += """## 结论

基于以上分析，建议优先处理高优先级问题，并建立长期的日志监控机制。

---
*报告由 Log Intelligence Multi-Agent System 自动生成*
"""
        
        return report
    
    def _format_analysis_section(self, analysis: Dict) -> str:
        """格式化分析部分"""
        analysis_type = analysis.get('type', 'unknown')
        
        type_names = {
            'error_analysis': '错误分析',
            'pattern_analysis': '模式分析',
            'trend_analysis': '趋势分析',
            'rag_analysis': '深度分析'
        }
        
        section = f"#### {type_names.get(analysis_type, analysis_type)}\n\n"
        
        if 'findings' in analysis and analysis['findings']:
            section += "**发现**:\n"
            for finding in analysis['findings']:
                if isinstance(finding, str):
                    section += f"- {finding}\n"
                elif isinstance(finding, dict):
                    section += f"- {finding.get('finding', finding)}\n"
            section += "\n"
        
        return section
    
    def _format_solution(self, solution: Dict) -> str:
        """格式化解决方案"""
        text = f"**{solution['title']}**\n\n"
        text += f"{solution.get('description', '')}\n\n"
        
        if solution.get('steps'):
            text += "**执行步骤**:\n"
            for i, step in enumerate(solution['steps'], 1):
                text += f"{i}. {step}\n"
            text += "\n"
        
        if solution.get('commands'):
            text += "**相关命令**:\n```bash\n"
            for cmd in solution['commands']:
                text += f"{cmd}\n"
            text += "```\n\n"
        
        text += f"*预估时间: {solution.get('estimated_time', '未知')} | 难度: {solution.get('difficulty', '未知')}*\n\n"
        text += "---\n\n"
        
        return text
    
    def _generate_json_report(self) -> str:
        """生成 JSON 报告"""
        import json
        return json.dumps(self.report_data, indent=2, ensure_ascii=False, default=str)
    
    def _generate_html_report(self) -> str:
        """生成 HTML 报告"""
        # 简化版 HTML
        md_content = self._generate_markdown_report()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>日志分析报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 20px; color: #666; }}
    </style>
</head>
<body>
    {self._markdown_to_html(md_content)}
</body>
</html>"""
        return html
    
    def _markdown_to_html(self, md: str) -> str:
        """简单的 Markdown 转 HTML"""
        import re
        
        html = md
        # 转换标题
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # 转换代码块
        html = re.sub(r'```bash\n(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        
        # 转换粗体
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        
        # 转换换行
        html = html.replace('\n\n', '</p><p>')
        
        return f'<p>{html}</p>'
    
    def _generate_summary(self) -> str:
        """生成执行摘要"""
        data = self.report_data
        
        summary_parts = []
        
        # 统计信息
        if data.get('parsed_data_summary'):
            summary = data['parsed_data_summary']
            summary_parts.append(f"分析了 {summary.get('total_entries', 0)} 条日志记录")
        
        # 问题数量
        if data.get('solutions'):
            high_count = len([s for s in data['solutions'] if s.get('priority') == 'high'])
            if high_count > 0:
                summary_parts.append(f"发现 {high_count} 个高优先级问题需要立即处理")
        
        # 建议
        if data.get('action_plan'):
            plan = data['action_plan']
            summary_parts.append(f"预计修复时间: {plan.get('estimated_total_time', '未知')}")
        
        return " | ".join(summary_parts) if summary_parts else "分析完成，未发现明显问题"
    
    def _extract_key_findings(self) -> List[str]:
        """提取关键发现"""
        return self.report_data.get('intermediate_findings', [])
    
    def _extract_next_steps(self, context: AgentContext) -> List[str]:
        """提取下一步行动"""
        steps = []
        
        action_plan = self.report_data.get('action_plan', {})
        if action_plan.get('recommended_sequence'):
            steps = action_plan['recommended_sequence'][:5]  # 前5个
        
        return steps
    
    def can_handle(self, context: AgentContext) -> bool:
        """总是可以处理（作为最后一步）"""
        return True
    
    def save_report(self, filepath: str, format_type: str = "markdown"):
        """保存报告到文件"""
        report = self._generate_report(format_type)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return filepath
