"""
核心架构模块:
- base:           基础抽象 (BaseAgent, AgentContext, AgentResult)
- orchestrator:   Agent 调度编排 (SEQUENTIAL / PARALLEL / CONDITIONAL)
- agent_router:   查询意图分类 + Agent 链路由 (AgentRouter)
- pipeline:       5 阶段流水线 (Fetch → Index → Process → Output → Evolve)
- router:         CLI 命令路由 (CommandRouter)
- factory:        工厂函数 (create_orchestrator, setup_environment)
- indexer:        混合检索 + RRF 融合
- scorer:         全链路打分卡
- protocol:       Agent 间消息总线 (MessageBus)

注: reflector (HallucinationGuard / ReflectionLoop) 已迁移至 financial_rag.guard
"""
from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, ExecutionMode
from .orchestrator import AgentOrchestrator, CoordinatorConfig, ExecutionResult
from .pipeline import PipelineScheduler, PipelineConfig as SchedulerPipelineConfig, PipelineResult as SchedulerPipelineResult
from .factory import create_orchestrator, create_hybrid_retriever, setup_environment
from .agent_router import AgentRouter, RoutingDecision, QueryIntent, create_agent_router
from .router import CommandRouter
from .indexer import PipelineOrchestrator, PipelineConfig, PipelineResult, PipelineStatus
from .scorer import PipelineScoreCard, StageScore, ScoreGrade, StageGroup, create_scorecard
from .protocol import AgentMessage, MessageBus, MessageAdapter
from .data_orchestrator import DataOrchestrator, KnowledgePool, DataRouter, IngestStats

# Backward-compat: reflector 已迁移至 financial_rag.guard，此处 re-export 避免旧代码报键
from financial_rag.guard.reflector import (  # noqa: F401
    ReflectionLoop, ReflectionConfig, ThoughtStep, ActionType, ReflectionState, HallucinationGuard,
)

__all__ = [
    # Base
    "BaseAgent", "AgentContext", "AgentResult", "AgentStatus", "ExecutionMode",
    # Orchestrator
    "AgentOrchestrator", "CoordinatorConfig", "ExecutionResult",
    # Pipeline Scheduler
    "PipelineScheduler", "SchedulerPipelineConfig", "SchedulerPipelineResult",
    # Factory
    "create_orchestrator", "create_hybrid_retriever", "setup_environment",
    # Agent Router
    "AgentRouter", "RoutingDecision", "QueryIntent", "create_agent_router",
    # CLI Router
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
    # Data Orchestrator
    "DataOrchestrator", "KnowledgePool", "DataRouter", "IngestStats",
]
