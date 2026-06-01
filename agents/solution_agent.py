"""
解决方案 Agent - 负责生成修复建议和解决方案
"""
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentContext, AgentResponse


class SolutionAgent(BaseAgent):
    """
    解决方案 Agent
    
    职责：
    1. 基于分析结果生成解决方案
    2. 提供逐步修复指导
    3. 评估方案优先级
    4. 生成可执行的命令或配置
    """
    
    def __init__(self):
        super().__init__(
            name="SolutionAgent",
            description="基于分析结果生成解决方案和修复建议"
        )
        self.solutions: List[Dict] = []
        self.knowledge_base = None
    
    def process(self, context: AgentContext) -> AgentResponse:
        """
        生成解决方案
        
        基于 context.analysis_results 生成具体解决方案
        """
        if not context.analysis_results:
            return AgentResponse(
                success=False,
                message="没有分析结果，无法生成解决方案"
            )
        
        # 设置知识库（如果有）
        if hasattr(context, 'knowledge_base') and context.metadata.get('knowledge_base'):
            self.knowledge_base = context.metadata['knowledge_base']
        
        # 生成解决方案
        self.solutions = self._generate_solutions(context)
        
        # 优先级排序
        self.solutions = self._prioritize_solutions(self.solutions)
        
        # 生成执行计划
        action_plan = self._create_action_plan()
        
        # 更新上下文
        context.metadata['solutions'] = self.solutions
        context.metadata['action_plan'] = action_plan
        
        # 添加到发现
        for sol in self.solutions[:3]:  # 只添加前3个
            context.add_finding(f"解决方案: {sol['title']} (优先级: {sol['priority']})")
        
        return AgentResponse(
            success=True,
            data={
                "solutions": self.solutions,
                "action_plan": action_plan,
                "immediate_actions": [s for s in self.solutions if s['priority'] == 'high']
            },
            message=f"生成 {len(self.solutions)} 个解决方案，建议优先处理 {len([s for s in self.solutions if s['priority'] == 'high'])} 个高优先级问题",
            context_updates={
                "solutions": self.solutions,
                "action_plan": action_plan
            }
        )
    
    def _generate_solutions(self, context: AgentContext) -> List[Dict]:
        """基于分析结果生成解决方案"""
        solutions = []
        analysis_results = context.analysis_results
        
        # 1. 处理错误分析结果
        if "analyses" in analysis_results:
            for analysis in analysis_results["analyses"]:
                sols = self._solutions_from_analysis(analysis, context)
                solutions.extend(sols)
        
        # 2. 处理建议
        if "recommendations" in analysis_results:
            for rec in analysis_results["recommendations"]:
                solutions.append({
                    "title": rec,
                    "description": rec,
                    "category": "general",
                    "priority": "medium",
                    "steps": [rec],
                    "estimated_time": "未知",
                    "difficulty": "medium"
                })
        
        # 3. 如果没有生成任何方案，提供通用建议
        if not solutions:
            solutions = self._generate_generic_solutions(context)
        
        return solutions
    
    def _solutions_from_analysis(self, analysis: Dict, context: AgentContext) -> List[Dict]:
        """从特定分析生成解决方案"""
        solutions = []
        analysis_type = analysis.get("type", "")
        
        if analysis_type == "error_analysis":
            # 错误相关的解决方案
            error_summary = analysis.get("error_summary", "")
            
            if "database" in error_summary.lower() or "connection" in error_summary.lower():
                solutions.append({
                    "title": "修复数据库连接问题",
                    "description": "检测到数据库连接异常，需要检查连接配置和网络",
                    "category": "database",
                    "priority": "high",
                    "steps": [
                        "检查数据库服务是否正常运行",
                        "验证连接字符串和凭据",
                        "检查网络连接和防火墙设置",
                        "查看数据库日志获取详细错误",
                        "考虑增加连接池大小或超时时间"
                    ],
                    "commands": [
                        "systemctl status postgresql",
                        "ping <database_host>",
                        "telnet <database_host> <port>"
                    ],
                    "estimated_time": "30-60分钟",
                    "difficulty": "medium"
                })
            
            if "memory" in error_summary.lower() or "oom" in error_summary.lower():
                solutions.append({
                    "title": "优化内存使用",
                    "description": "系统内存使用过高，可能导致 OOM",
                    "category": "performance",
                    "priority": "high",
                    "steps": [
                        "检查内存泄漏（使用 top/htop）",
                        "分析内存占用最高的进程",
                        "调整应用内存限制",
                        "考虑增加物理内存或 SWAP",
                        "优化缓存策略"
                    ],
                    "commands": [
                        "free -h",
                        "ps aux --sort=-%mem | head -20",
                        "jmap -heap <pid>  # Java应用"
                    ],
                    "estimated_time": "1-2小时",
                    "difficulty": "hard"
                })
            
            if "timeout" in error_summary.lower():
                solutions.append({
                    "title": "解决超时问题",
                    "description": "检测到多个超时错误",
                    "category": "network",
                    "priority": "medium",
                    "steps": [
                        "检查网络延迟",
                        "增加超时配置",
                        "优化慢查询或慢接口",
                        "检查依赖服务健康状态"
                    ],
                    "estimated_time": "30-60分钟",
                    "difficulty": "medium"
                })
        
        elif analysis_type == "pattern_analysis":
            patterns = analysis.get("patterns", {})
            
            if patterns.get("repeated_errors"):
                solutions.append({
                    "title": "处理重复错误模式",
                    "description": "发现重复发生的错误，需要系统性修复",
                    "category": "bugfix",
                    "priority": "high",
                    "steps": [
                        "分析错误发生的共同条件",
                        "定位问题代码或配置",
                        "实施修复",
                        "添加监控和告警",
                        "验证修复效果"
                    ],
                    "estimated_time": "2-4小时",
                    "difficulty": "hard"
                })
        
        elif analysis_type == "trend_analysis":
            peak_hours = analysis.get("peak_error_hours", [])
            if peak_hours:
                solutions.append({
                    "title": "优化高峰期性能",
                    "description": f"错误集中在 {peak_hours[0]['hour']}",
                    "category": "performance",
                    "priority": "medium",
                    "steps": [
                        "分析高峰期的流量模式",
                        "检查资源使用情况（CPU/内存/IO）",
                        "考虑扩容或限流",
                        "优化热点代码"
                    ],
                    "estimated_time": "2-6小时",
                    "difficulty": "hard"
                })
        
        return solutions
    
    def _generate_generic_solutions(self, context: AgentContext) -> List[Dict]:
        """生成通用解决方案"""
        return [
            {
                "title": "审查系统日志",
                "description": "定期检查系统日志，及时发现潜在问题",
                "category": "maintenance",
                "priority": "low",
                "steps": [
                    "设置日志轮转",
                    "配置日志监控告警",
                    "建立日志审查流程"
                ],
                "estimated_time": "1小时",
                "difficulty": "easy"
            },
            {
                "title": "更新文档",
                "description": "将本次分析结果记录到运维文档",
                "category": "documentation",
                "priority": "low",
                "steps": [
                    "记录发现的问题",
                    "更新故障排查手册",
                    "分享给团队成员"
                ],
                "estimated_time": "30分钟",
                "difficulty": "easy"
            }
        ]
    
    def _prioritize_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """按优先级排序"""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(solutions, key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
    
    def _create_action_plan(self) -> Dict:
        """创建执行计划"""
        high_priority = [s for s in self.solutions if s['priority'] == 'high']
        medium_priority = [s for s in self.solutions if s['priority'] == 'medium']
        low_priority = [s for s in self.solutions if s['priority'] == 'low']
        
        return {
            "immediate": high_priority,
            "short_term": medium_priority,
            "long_term": low_priority,
            "estimated_total_time": self._estimate_total_time(),
            "recommended_sequence": self._recommend_sequence()
        }
    
    def _estimate_total_time(self) -> str:
        """估算总时间"""
        # 简化估算
        high_count = len([s for s in self.solutions if s['priority'] == 'high'])
        medium_count = len([s for s in self.solutions if s['priority'] == 'medium'])
        
        total_hours = high_count * 1.5 + medium_count * 1
        return f"约 {total_hours:.1f} 小时"
    
    def _recommend_sequence(self) -> List[str]:
        """推荐执行顺序"""
        sequence = []
        
        # 先处理高优先级
        for sol in self.solutions:
            if sol['priority'] == 'high':
                sequence.append(f"[高优先级] {sol['title']}")
        
        # 再处理中优先级
        for sol in self.solutions:
            if sol['priority'] == 'medium':
                sequence.append(f"[中优先级] {sol['title']}")
        
        return sequence
    
    def can_handle(self, context: AgentContext) -> bool:
        """需要分析结果"""
        return bool(context.analysis_results)
    
    def get_solution_by_category(self, category: str) -> List[Dict]:
        """按类别获取解决方案"""
        return [s for s in self.solutions if s.get('category') == category]
    
    def get_immediate_actions(self) -> List[Dict]:
        """获取立即执行的方案"""
        return [s for s in self.solutions if s.get('priority') == 'high']
