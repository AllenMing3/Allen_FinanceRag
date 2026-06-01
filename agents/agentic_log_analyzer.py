"""
Agentic RAG 日志分析 Agent - 改造后的智能日志分析

改造要点：
1. 从"一次检索"改为"多次检索"
2. 添加自我反思机制
3. 支持查询优化
4. 完整的 ReAct 循环展示

对比传统 Agentic RAG：
- 传统 Agent：一次检索 -> 直接回答
- Agentic Agent：检索 -> 反思 -> 优化查询 -> 再检索 -> 综合 -> 判断质量
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .agentic_base_agent import (
    BaseAgenticAgent, AgenticRAGConfig, AgenticState,
    ActionType, ThoughtStep, RetrievalContext
)
from .base_agent import AgentContext, AgentResponse
from .scoring_system import ScoringSystem, SimpleScoringSystem
from knowledge_base import LogKnowledgeBase
from rag_engine import RAGEngine, QueryType
import json


class AgenticLogAnalyzerConfig(AgenticRAGConfig):
    """日志分析 Agent 的专用配置"""
    def __init__(self):
        super().__init__()
        # 检索配置
        self.max_retrievals = 4              # 日志分析需要更多检索
        self.retrieval_top_k = 8             # 返回更多候选结果

        # 查询优化
        self.enable_query_rewrite = True
        self.query_variations = [
            "日志中的错误和异常",
            "系统性能和资源问题",
            "业务逻辑和业务流程问题",
            "配置和环境相关问题"
        ]

        # 自我反思
        self.enable_self_reflection = True
        self.min_confidence_threshold = 0.65

        # 专门的检索策略
        self.retrieval_strategies = [
            {"type": "error", "weight": 0.4, "description": "错误相关检索"},
            {"type": "pattern", "weight": 0.3, "description": "模式识别检索"},
            {"type": "solution", "weight": 0.3, "description": "解决方案检索"}
        ]


class AgenticLogAnalyzer(BaseAgenticAgent):
    """
    Agentic RAG 日志分析器

    核心能力：
    1. 多轮检索：不是一次检索就完事，而是根据反思结果决定是否需要更多检索
    2. 查询优化：针对不同方面生成不同查询
    3. 自我反思：评估当前回答是否足够好
    4. 综合推理：将多次检索结果综合成完整分析
    """

    def __init__(self, config: Optional[AgenticLogAnalyzerConfig] = None):
        super().__init__(
            name="AgenticLogAnalyzer",
            description="使用 Agentic RAG 进行智能日志分析",
            config=config or AgenticLogAnalyzerConfig()
        )

        self.knowledge_base: Optional[LogKnowledgeBase] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.current_analysis_focus: str = ""  # 当前分析重点
        
        # 新的评分系统，解决评分标准上升太快的问题
        self.scoring_system = SimpleScoringSystem()
        self.full_scoring_system = ScoringSystem()

    def _determine_task(self, context: AgentContext) -> str:
        """从上下文确定任务"""
        if context.raw_input:
            return f"分析以下日志内容：{context.raw_input[:200]}..."

        if context.parsed_data:
            entry_count = len(context.parsed_data) if isinstance(context.parsed_data, list) else 0
            return f"分析 {entry_count} 条日志条目，识别问题并提供解决方案"

        return "分析日志，识别错误、警告和异常模式"

    def _think(self, step: int) -> ThoughtStep:
        """
        Think 阶段：分析日志的思考逻辑

        改造后的思考策略：
        - 步骤1：检索错误相关信息
        - 步骤2：检索模式相关信息
        - 步骤3：检索解决方案
        - 步骤4：综合分析
        """
        if not self.state:
            return ThoughtStep(step_number=step, thought="初始化失败", action=ActionType.FINISH, action_input="")

        retrieval_count = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)

        # 根据步骤决定检索策略
        if retrieval_count == 0:
            # 第一次检索：错误信息
            self.current_analysis_focus = "error"
            return ThoughtStep(
                step_number=step,
                thought="首先检索日志中的错误和异常信息，了解主要问题",
                action=ActionType.RETRIEVE,
                action_input=self._build_error_query()
            )
        elif retrieval_count == 1:
            # 第二次检索：模式分析
            self.current_analysis_focus = "pattern"
            return ThoughtStep(
                step_number=step,
                thought="基于第一次检索结果，现在检索相关的错误模式和趋势",
                action=ActionType.RETRIEVE,
                action_input=self._build_pattern_query()
            )
        elif retrieval_count == 2:
            # 第三次检索：解决方案
            self.current_analysis_focus = "solution"
            return ThoughtStep(
                step_number=step,
                thought="检索相关的解决方案和修复建议",
                action=ActionType.RETRIEVE,
                action_input=self._build_solution_query()
            )
        elif retrieval_count >= 3:
            # 检查是否需要更多检索
            if self._needs_more_retrieval():
                return ThoughtStep(
                    step_number=step,
                    thought="评估发现信息不够完整，尝试从其他角度检索",
                    action=ActionType.RETRIEVE,
                    action_input=self._build_additional_query()
                )

            # 综合答案
            return ThoughtStep(
                step_number=step,
                thought="已有足够信息，进行综合分析",
                action=ActionType.SYNTHESIZE,
                action_input="generate_comprehensive_analysis"
            )

        return ThoughtStep(step_number=step, thought="继续", action=ActionType.FINISH, action_input="")

    def _build_error_query(self) -> str:
        """构建错误检索查询"""
        return "日志中的错误、异常、失败信息，关键错误原因和错误堆栈"

    def _build_pattern_query(self) -> str:
        """构建模式检索查询"""
        return "错误模式、错误频率、相关日志事件、错误时间分布"

    def _build_solution_query(self) -> str:
        """构建解决方案检索查询"""
        return "错误解决方法、修复建议、配置调整、系统优化建议"

    def _build_additional_query(self) -> str:
        """构建补充检索查询"""
        return "系统性能、资源使用、业务流程、配置问题"

    def _needs_more_retrieval(self) -> bool:
        """判断是否需要更多检索"""
        if not self.state or not self.state.retrieved_contexts:
            return True

        # 检查置信度
        if self.state.confidence < 0.5:
            return True

        # 检查结果数量
        total_results = sum(len(ctx.results) for ctx in self.state.retrieved_contexts)
        if total_results < 5:
            return True

        return False

    def _do_retrieve(self, query: str) -> str:
        """
        执行检索动作

        这是 Agentic RAG 的核心：不是简单调用一次检索，
        而是根据 query 类型选择合适的检索策略
        """
        if not self.rag_engine:
            return "知识库未初始化，无法检索"

        # 根据分析重点选择查询类型
        if self.current_analysis_focus == "error":
            query_type = QueryType.TROUBLESHOOT
        elif self.current_analysis_focus == "pattern":
            query_type = QueryType.ANALYSIS
        elif self.current_analysis_focus == "solution":
            query_type = QueryType.TROUBLESHOOT
        else:
            query_type = QueryType.GENERAL

        try:
            # 执行检索
            result = self.rag_engine.query(query, query_type=query_type)

            # 保存检索结果
            self.state.add_retrieval(query, result.sources)

            # 构建观察结果描述
            observation = f"检索到 {len(result.sources)} 条相关信息，"
            observation += f"置信度 {result.confidence:.2f}\n\n"

            if result.sources:
                # 展示前3个最相关的结果
                for i, source in enumerate(result.sources[:3], 1):
                    observation += f"结果{i}: {source.get('text', '')[:150]}...\n\n"

            return observation

        except Exception as e:
            return f"检索失败: {str(e)}"

    def _do_synthesize(self) -> str:
        """
        执行综合动作

        这是 Agentic RAG 的关键：将多次检索结果综合成完整分析
        """
        context_str = self.state.get_all_context()

        # 使用 LLM 综合答案
        self.state.synthesis = self._synthesize_with_llm(
            task=self.state.task,
            context=context_str,
            focus=self.current_analysis_focus
        )

        # 评估置信度
        self.state.confidence = self._assess_confidence()

        return self.state.synthesis

    def _synthesize_with_llm(self, task: str, context: str, focus: str = "") -> str:
        """
        使用 LLM 综合答案

        实际项目中应该调用 Qwen/DeepSeek 等 LLM
        这里提供一个模板，实际使用时替换为真正的 LLM 调用
        """
        # 分析检索到的内容
        error_patterns = []
        solutions = []
        sources = []

        for ctx in self.state.retrieved_contexts:
            for result in ctx.results:
                text = result.get('text', '')
                score = result.get('score', 0)

                if score < 0.3:  # 过滤低相关度结果
                    continue

                # 简单的分类（实际应该用 LLM）
                if '解决' in text or '修复' in text or '方案' in text:
                    solutions.append(text)
                else:
                    error_patterns.append(text)

                sources.append(result)

        # 构建综合报告
        synthesis = self._build_analysis_report(error_patterns, solutions, sources)

        return synthesis

    def _build_analysis_report(self, error_patterns: List[str], solutions: List[str], sources: List[Dict]) -> str:
        """构建分析报告"""
        report_parts = []

        # 1. 执行摘要
        report_parts.append("=" * 50)
        report_parts.append("📊 日志分析报告 (Agentic RAG)")
        report_parts.append("=" * 50)

        retrieval_count = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)
        report_parts.append(f"\n检索次数: {retrieval_count}")
        report_parts.append(f"信息来源: {len(sources)} 条")
        report_parts.append(f"置信度: {self.state.confidence:.2f}")

        # 2. 问题分析
        report_parts.append("\n\n## 🔍 问题分析")
        report_parts.append("-" * 30)

        if error_patterns:
            report_parts.append(f"发现 {len(set(error_patterns))} 个潜在问题：\n")
            for i, pattern in enumerate(error_patterns[:5], 1):
                report_parts.append(f"{i}. {pattern[:200]}...")
        else:
            report_parts.append("未发现明显错误模式")

        # 3. 解决方案
        report_parts.append("\n\n## 💡 解决方案")
        report_parts.append("-" * 30)

        if solutions:
            for i, sol in enumerate(solutions[:5], 1):
                report_parts.append(f"{i}. {sol[:200]}...")
        else:
            report_parts.append("基于检索结果，建议：")
            report_parts.append("  - 检查系统日志中的错误堆栈")
            report_parts.append("  - 关注高频错误模式")
            report_parts.append("  - 审查最近的系统变更")

        # 4. 置信度说明
        report_parts.append("\n\n## 📋 置信度说明")
        report_parts.append("-" * 30)

        if self.state.confidence >= 0.7:
            report_parts.append("✅ 高置信度：检索结果与问题高度相关，分析可靠")
        elif self.state.confidence >= 0.5:
            report_parts.append("⚠️ 中置信度：部分检索结果相关，建议进一步验证")
        else:
            report_parts.append("⚠️ 低置信度：检索结果相关性有限，建议人工审查")

        return "\n".join(report_parts)

    def _assess_confidence(self) -> float:
        """
        评估置信度

        考虑因素：
        1. 检索次数
        2. 检索结果数量
        3. 结果相关度分数
        4. 是否包含解决方案
        """
        if not self.state or not self.state.retrieved_contexts:
            return 0.0

        confidence = 0.2  # 基础置信度

        # 检索次数加成（最多 +0.3）
        retrieval_count = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)
        confidence += min(0.3, retrieval_count * 0.1)

        # 结果数量加成（最多 +0.2）
        total_results = sum(len(ctx.results) for ctx in self.state.retrieved_contexts)
        confidence += min(0.2, total_results * 0.02)

        # 结果质量加成（最多 +0.3）
        all_scores = []
        for ctx in self.state.retrieved_contexts:
            all_scores.extend(ctx.results.get('score', 0) for r in ctx.results if 'score' in r)

        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            confidence += min(0.3, avg_score * 0.4)

        return min(0.95, max(0.1, confidence))

    def _judge(self, thought: ThoughtStep) -> tuple[bool, str]:
        """
        Judge 阶段：评估当前状态

        Agentic RAG 的自我反思机制
        """
        # 如果刚完成综合
        if thought.action == ActionType.SYNTHESIZE and self.state.synthesis:
            if self.state.confidence >= self.config.min_confidence_threshold:
                return False, f"置信度达标 ({self.state.confidence:.2f} >= {self.config.min_confidence_threshold})，停止"

            # 检查是否已达到最大检索次数
            retrieval_count = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)
            if retrieval_count >= self.config.max_retrievals:
                return False, "达到最大检索次数，强制停止"

        # 检查是否达到最大步数
        if thought.step_number >= self.config.max_steps:
            return False, "达到最大步数限制"

        # 检查是否需要更多检索
        if thought.action == ActionType.SYNTHESIZE:
            retrieval_count = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)
            if retrieval_count < self.config.max_retrievals and self.state.confidence < 0.5:
                return True, "置信度不足，继续检索"

        return True, "继续执行"

    def process(self, context: AgentContext) -> AgentResponse:
        """
        处理日志分析请求

        这是与 BaseAgent 接口的兼容方法
        """
        # 初始化知识库（如果需要）
        if not self.knowledge_base and context.parsed_data:
            self._init_knowledge_base(context)

        if not self.rag_engine:
            return AgentResponse(
                success=False,
                message="RAG 引擎未初始化，请先提供日志数据",
                agent_name=self.name
            )

        # 运行 Agentic RAG
        return self.run_agentic(context)

    def _init_knowledge_base(self, context: AgentContext):
        """初始化知识库"""
        from log_parser import LogEntry

        self.knowledge_base = LogKnowledgeBase()

        # 过滤有效条目
        entries = [e for e in context.parsed_data if isinstance(e, LogEntry)]

        if entries:
            self.knowledge_base.add_logs(entries).build_index()
            self.rag_engine = RAGEngine(self.knowledge_base)

            context.add_finding(f"已构建知识库，包含 {len(entries)} 条日志")

    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理"""
        return context.parsed_data is not None or context.raw_input is not None


def demo_agentic_rag():
    """
    Agentic RAG 演示

    展示 ReAct 循环的执行过程
    """
    print("\n" + "=" * 70)
    print("Agentic RAG 演示 - ReAct 循环")
    print("=" * 70)

    # 创建 Agentic Agent
    analyzer = AgenticLogAnalyzer()

    # 模拟简单的上下文
    context = AgentContext(
        raw_input="示例日志：ERROR: 连接超时，WARN: 内存使用率 85%，INFO: 用户登录成功"
    )

    # 运行
    response = analyzer.run_agentic(context)

    # 展示思考链
    print("\n\n" + "=" * 70)
    print("完整 ReAct 循环")
    print("=" * 70)
    print(analyzer.get_thought_chain())

    return response


if __name__ == "__main__":
    demo_agentic_rag()
