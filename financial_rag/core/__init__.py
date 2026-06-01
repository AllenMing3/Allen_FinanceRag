"""
核心架构三大件:
- Coordinate: 多 Agent 协调调度
- Indexer: 多文本索引流水线
- Reflector: ReAct 反思 + 六层防幻觉
"""
from .coordinator import AgentOrchestrator, CoordinatorConfig, ExecutionMode, ExecutionResult
from .indexer import PipelineOrchestrator, PipelineConfig, PipelineResult, PipelineStatus
from .reflector import ReflectionLoop, ReflectionConfig, ThoughtStep, ActionType, ReflectionState, HallucinationGuard

__all__ = [
    # Coordinate
    "AgentOrchestrator", "CoordinatorConfig", "ExecutionMode", "ExecutionResult",
    # Indexer
    "PipelineOrchestrator", "PipelineConfig", "PipelineResult", "PipelineStatus",
    # Reflector
    "ReflectionLoop", "ReflectionConfig", "ThoughtStep", "ActionType", "ReflectionState",
    "HallucinationGuard",
]
