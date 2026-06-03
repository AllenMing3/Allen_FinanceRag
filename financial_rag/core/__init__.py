"""
核心架构模块:
- base:        基础抽象 (BaseAgent, AgentContext, AgentResult)
- orchestrator: Agent 调度编排 (SEQUENTIAL / PARALLEL / CONDITIONAL)
- pipeline:    5 阶段流水线 (Fetch → Index → Process → Output → Evolve)
- router:      CLI 命令路由 (CommandRouter)
- factory:     工厂函数 (create_orchestrator, setup_environment)
- indexer:     混合检索 + RRF 融合
- reflector:   ReAct 反思 + 防幻觉
- scorer:      全链路打分卡
- protocol:    Agent 间消息总线 (MessageBus)
"""
from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, ExecutionMode
from .orchestrator import AgentOrchestrator, CoordinatorConfig, ExecutionResult
from .pipeline import PipelineScheduler, PipelineConfig as SchedulerPipelineConfig, PipelineResult as SchedulerPipelineResult
from .factory import create_orchestrator, create_hybrid_retriever, setup_environment
from .router import CommandRouter
from .indexer import PipelineOrchestrator, PipelineConfig, PipelineResult, PipelineStatus
from .reflector import ReflectionLoop, ReflectionConfig, ThoughtStep, ActionType, ReflectionState, HallucinationGuard
from .scorer import PipelineScoreCard, StageScore, ScoreGrade, StageGroup, create_scorecard
from .protocol import AgentMessage, MessageBus, MessageAdapter

__all__ = [
    # Base
    "BaseAgent", "AgentContext", "AgentResult", "AgentStatus", "ExecutionMode",
    # Orchestrator
    "AgentOrchestrator", "CoordinatorConfig", "ExecutionResult",
    # Pipeline Scheduler
    "PipelineScheduler", "SchedulerPipelineConfig", "SchedulerPipelineResult",
    # Factory
    "create_orchestrator", "create_hybrid_retriever", "setup_environment",
    # Router
    "CommandRouter",
    # Indexer
    "PipelineOrchestrator", "PipelineConfig", "PipelineResult", "PipelineStatus",
    # Reflector
    "ReflectionLoop", "ReflectionConfig", "ThoughtStep", "ActionType", "ReflectionState",
    "HallucinationGuard",
    # Scorer
    "PipelineScoreCard", "StageScore", "ScoreGrade", "StageGroup", "create_scorecard",
    # Protocol
    "AgentMessage", "MessageBus", "MessageAdapter",
]
