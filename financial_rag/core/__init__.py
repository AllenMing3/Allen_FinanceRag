"""
核心架构三大件 + 全链路打分:
- Coordinate: 多 Agent 协调调度
- Indexer: 多文本索引流水线
- Reflector: ReAct 反思 + 六层防幻觉
- Scorer: 全链路打分卡（每阶段独立评分 + 诊断）
"""
from .coordinator import AgentOrchestrator, CoordinatorConfig, ExecutionMode, ExecutionResult
from .indexer import PipelineOrchestrator, PipelineConfig, PipelineResult, PipelineStatus
from .reflector import ReflectionLoop, ReflectionConfig, ThoughtStep, ActionType, ReflectionState, HallucinationGuard
from .scorer import PipelineScoreCard, StageScore, ScoreGrade, StageGroup, create_scorecard

__all__ = [
    # Coordinate
    "AgentOrchestrator", "CoordinatorConfig", "ExecutionMode", "ExecutionResult",
    # Indexer
    "PipelineOrchestrator", "PipelineConfig", "PipelineResult", "PipelineStatus",
    # Reflector
    "ReflectionLoop", "ReflectionConfig", "ThoughtStep", "ActionType", "ReflectionState",
    "HallucinationGuard",
    # Scorer
    "PipelineScoreCard", "StageScore", "ScoreGrade", "StageGroup", "create_scorecard",
]
