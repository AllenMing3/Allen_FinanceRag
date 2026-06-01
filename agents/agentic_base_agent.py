"""
Agentic RAG Agent 基类 - 带有 ReAct 循环的智能 Agent

核心概念：
1. ReAct 循环 = Think + Retrieve + Act + Observe + Judge
2. 查询优化：根据检索结果优化查询策略
3. 自我反思：评估回答质量，决定是否需要重新检索
4. 多次检索：支持在同一任务中多次检索不同方面的信息
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time

from .base_agent import BaseAgent, AgentContext, AgentResponse, AgentStatus


class ActionType(Enum):
    """Agent 可执行的动作类型"""
    RETRIEVE = "retrieve"           # 检索知识库
    REFINE_QUERY = "refine_query"   # 优化查询
    ANALYZE = "analyze"            # 分析数据
    SYNTHESIZE = "synthesize"      # 综合答案
    FINISH = "finish"              # 完成
    RETRY = "retry"                # 重试


@dataclass
class ThoughtStep:
    """ReAct 循环中的单步思考"""
    step_number: int
    thought: str                    # 思考：当前状态是什么？
    action: ActionType              # 动作：要做什么？
    action_input: str               # 动作输入：具体参数
    observation: str = ""           # 观察：动作的结果
    reflection: str = ""            # 反思：这个结果是否有帮助？

    def to_dict(self) -> Dict:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action.value,
            "action_input": self.action_input,
            "observation": self.observation,
            "reflection": self.reflection
        }


@dataclass
class RetrievalContext:
    """检索上下文"""
    query: str
    results: List[Dict] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    query_variations: List[str] = field(default_factory=list)
    retrieval_count: int = 0

    def add_result(self, query: str, results: List[Dict]):
        """添加检索结果"""
        self.query = query
        self.results.extend(results)
        self.retrieval_count += 1
        if query not in self.query_variations:
            self.query_variations.append(query)

    def get_context_str(self, max_chars: int = 4000) -> str:
        """将检索结果格式化为上下文字符串"""
        if not self.results:
            return "无可用检索结果"

        context_parts = []
        current_len = 0

        for result in self.results:
            text = result.get('text', '')[:500]  # 截断长文本
            score = result.get('score', 0)
            source = result.get('metadata', {}).get('source', 'unknown')

            part = f"[来源: {source}, 相关度: {score:.2f}]\n{text}\n"
            if current_len + len(part) <= max_chars:
                context_parts.append(part)
                current_len += len(part)
            else:
                break

        return "\n---\n".join(context_parts)


@dataclass
class AgenticState:
    """Agentic Agent 的内部状态"""
    task: str = ""                  # 当前任务
    current_query: str = ""         # 当前查询
    retrieved_contexts: List[RetrievalContext] = field(default_factory=list)
    synthesis: str = ""             # 综合的答案
    confidence: float = 0.0         # 置信度
    should_continue: bool = True    # 是否继续循环
    reason_for_continuation: str = "" # 继续的原因
    max_steps_reached: bool = False
    quality_assessment: str = ""     # 质量评估

    def add_retrieval(self, query: str, results: List[Dict]):
        """添加检索结果"""
        if not self.retrieved_contexts:
            self.retrieved_contexts.append(RetrievalContext(query=query, results=results))
        else:
            self.retrieved_contexts[-1].add_result(query, results)

    def get_all_context(self) -> str:
        """获取所有检索上下文"""
        return "\n\n".join([
            ctx.get_context_str() for ctx in self.retrieved_contexts
        ])


class AgenticRAGConfig:
    """Agentic RAG 配置"""
    def __init__(self):
        # 检索配置
        self.max_retrievals: int = 3          # 最多检索次数
        self.retrieval_top_k: int = 5         # 每次检索返回数量
        self.min_relevance_score: float = 0.3  # 最低相关度阈值

        # ReAct 循环配置
        self.max_steps: int = 6               # 最大循环步数
        self.think_before_action: bool = True # 执行前先思考

        # 查询优化配置
        self.enable_query_rewrite: bool = True  # 启用查询改写
        self.query_variations: List[str] = [
            "直接查询",
            "从错误角度查询",
            "从解决方案角度查询"
        ]

        # 自我反思配置
        self.enable_self_reflection: bool = True  # 启用自我反思
        self.min_confidence_threshold: float = 0.6  # 最低置信度阈值
        self.max_retries: int = 2              # 最多重试次数

        # LLM 配置
        self.llm_model: str = "gpt-4"
        self.temperature: float = 0.0


class BaseAgenticAgent(BaseAgent):
    """
    Agentic RAG Agent 基类

    特点：
    1. 内置 ReAct 循环
    2. 支持多次检索和查询优化
    3. 内置自我反思机制
    4. 可配置的检索策略
    """

    def __init__(self, name: str, description: str = "", config: Optional[AgenticRAGConfig] = None):
        super().__init__(name, description)
        self.config = config or AgenticRAGConfig()
        self.state: Optional[AgenticState] = None
        self.thought_history: List[ThoughtStep] = []

    def _init_state(self, task: str):
        """初始化 Agent 状态"""
        self.state = AgenticState(task=task)
        self.thought_history = []

    # ==================== 核心 ReAct 循环 ====================

    def run_agentic(self, context: AgentContext) -> AgentResponse:
        """
        运行 Agentic RAG Agent 的主入口

        ReAct 循环流程：
        1. Think - 分析当前状态，决定下一步
        2. Retrieve - 必要时检索知识库
        3. Act - 执行动作（分析、综合等）
        4. Observe - 观察结果
        5. Judge - 反思结果质量，决定是否继续
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING

        try:
            # 初始化状态
            task = self._determine_task(context)
            self._init_state(task)

            print(f"\n{'='*60}")
            print(f"[{self.name}] Agentic RAG 启动")
            print(f"任务: {task}")
            print(f"{'='*60}\n")

            # 执行 ReAct 循环
            step = 0
            while self.state.should_continue and step < self.config.max_steps:
                step += 1
                print(f"\n--- 步骤 {step}/{self.config.max_steps} ---")

                # 1. Think - 分析状态
                thought = self._think(step)

                # 2. 决定动作
                action, action_input = self._decide_action(thought)

                print(f"思考: {thought.thought}")
                print(f"动作: {action.value} -> {action_input}")

                # 3. 执行动作
                observation = self._execute_action(action, action_input)

                # 4. 更新观察结果
                thought.observation = observation
                self.thought_history.append(thought)

                # 5. Judge - 反思
                should_continue, reason = self._judge(thought)

                self.state.should_continue = should_continue
                self.state.reason_for_continuation = reason

                print(f"观察: {observation[:100]}..." if len(observation) > 100 else f"观察: {observation}")
                print(f"反思: {reason}")

            # 最终综合
            final_answer = self._synthesize()

            execution_time = time.time() - start_time

            # 构建响应
            response = AgentResponse(
                success=True,
                data={
                    "answer": final_answer,
                    "confidence": self.state.confidence,
                    "retrieval_count": sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts),
                    "steps": len(self.thought_history),
                    "thought_history": [t.to_dict() for t in self.thought_history]
                },
                message=f"完成，置信度: {self.state.confidence:.2f}, 检索次数: {self.state.retrieval_count}",
                agent_name=self.name,
                execution_time=execution_time,
                context_updates={"agentic_result": final_answer}
            )

            self.status = AgentStatus.COMPLETED
            return response

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResponse(
                success=False,
                message=f"执行失败: {str(e)}",
                agent_name=self.name,
                execution_time=time.time() - start_time
            )

    def _think(self, step: int) -> ThoughtStep:
        """
        Think 阶段：分析当前状态，决定下一步

        子类可重写此方法实现自定义思考逻辑
        """
        # 检查是否已有足够信息
        if self.state.synthesis and self.state.confidence >= self.config.min_confidence_threshold:
            return ThoughtStep(
                step_number=step,
                thought="已获得足够信息，可以进行综合",
                action=ActionType.SYNTHESIZE,
                action_input="generate_answer"
            )

        # 检查是否需要更多检索
        if not self.state.retrieved_contexts or step <= len(self.state.retrieved_contexts) + 1:
            return ThoughtStep(
                step_number=step,
                thought=self._generate_retrieval_thought(step),
                action=ActionType.RETRIEVE,
                action_input=self._generate_retrieval_query()
            )

        # 默认进行综合
        return ThoughtStep(
            step_number=step,
            thought="尝试综合已有信息生成回答",
            action=ActionType.SYNTHESIZE,
            action_input="generate_answer"
        )

    def _generate_retrieval_thought(self, step: int) -> str:
        """生成检索思考（子类可重写）"""
        return f"需要检索更多相关信息来回答问题"

    def _generate_retrieval_query(self) -> str:
        """生成检索查询（子类可重写）"""
        if not self.state.current_query:
            self.state.current_query = self.state.task
        return self.state.current_query

    def _decide_action(self, thought: ThoughtStep) -> tuple[ActionType, str]:
        """决定动作"""
        return thought.action, thought.action_input

    def _execute_action(self, action: ActionType, action_input: str) -> str:
        """
        执行动作

        子类必须实现具体的动作执行逻辑
        """
        if action == ActionType.RETRIEVE:
            return self._do_retrieve(action_input)
        elif action == ActionType.SYNTHESIZE:
            return self._do_synthesize()
        elif action == ActionType.FINISH:
            return self.state.synthesis
        else:
            return "未知动作类型"

    def _judge(self, thought: ThoughtStep) -> tuple[bool, str]:
        """
        Judge 阶段：评估当前状态，决定是否继续

        子类可重写此方法实现自定义的停止条件
        """
        # 如果已经综合了答案
        if thought.action == ActionType.SYNTHESIZE and self.state.synthesis:
            # 评估置信度
            if self.state.confidence >= self.config.min_confidence_threshold:
                return False, f"置信度 {self.state.confidence:.2f} 已达标，停止"
            elif thought.step_number >= self.config.max_steps:
                return False, "达到最大步数限制，停止"

        # 如果检索次数过多
        total_retrievals = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)
        if total_retrievals >= self.config.max_retrievals:
            return False, f"检索次数已达上限 ({total_retrievals})"

        # 如果步数达到上限
        if thought.step_number >= self.config.max_steps:
            self.state.max_steps_reached = True
            return False, "达到最大步数限制"

        return True, "继续执行"

    # ==================== 可被子类重写的方法 ====================

    def _determine_task(self, context: AgentContext) -> str:
        """从上下文确定任务"""
        return context.raw_input or self.description

    @abstractmethod
    def _do_retrieve(self, query: str) -> str:
        """
        执行检索动作

        子类必须实现
        """
        pass

    def _do_synthesize(self) -> str:
        """
        执行综合动作

        子类可重写
        """
        context_str = self.state.get_all_context()

        # 简单的综合（子类可以用更强的 LLM）
        self.state.synthesis = self._synthesize_with_llm(
            task=self.state.task,
            context=context_str
        )

        # 评估置信度
        self.state.confidence = self._assess_confidence()

        return self.state.synthesis

    def _synthesize_with_llm(self, task: str, context: str) -> str:
        """
        使用 LLM 综合答案

        子类可重写使用自己的 LLM 实现
        """
        # 简单实现（实际应该调用 LLM）
        return f"基于 {len(self.state.retrieved_contexts)} 次检索结果，综合回答：\n\n{context[:500]}...\n\n（此处应调用 LLM 生成完整回答）"

    def _assess_confidence(self) -> float:
        """
        评估置信度

        子类可重写
        """
        # 基于检索结果数量和质量评估
        retrieval_count = sum(ctx.retrieval_count for ctx in self.state.retrieved_contexts)
        result_count = sum(len(ctx.results) for ctx in self.state.retrieved_contexts)

        # 简单置信度计算
        confidence = 0.3
        confidence += min(0.2, retrieval_count * 0.1)
        confidence += min(0.3, result_count * 0.05)

        return min(0.95, confidence)

    def get_thought_chain(self) -> str:
        """获取思考链（用于调试和展示）"""
        lines = []
        for step in self.thought_history:
            lines.append(f"\n步骤 {step.step_number}:")
            lines.append(f"  思考: {step.thought}")
            lines.append(f"  动作: {step.action.value} -> {step.action_input}")
            lines.append(f"  观察: {step.observation[:100]}..." if len(step.observation) > 100 else f"  观察: {step.observation}")
            lines.append(f"  反思: {step.reflection}")
        return "\n".join(lines)
