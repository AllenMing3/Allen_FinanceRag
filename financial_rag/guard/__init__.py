"""
financial_rag.guard — 防幻觉与反思层

独立模块，与 core/ 平级，职责清晰：
- HallucinationGuard: 六层递进式防幻觉校验
- ReflectionLoop: ReAct 反思循环引擎

用法:
    from financial_rag.guard import HallucinationGuard
    from financial_rag.guard.reflector import ReflectionLoop, ReflectionConfig
"""
from .reflector import (
    HallucinationGuard,
    ReflectionLoop,
    ReflectionConfig,
    ReflectionState,
    ThoughtStep,
    ActionType,
)

__all__ = [
    "HallucinationGuard",
    "ReflectionLoop",
    "ReflectionConfig",
    "ReflectionState",
    "ThoughtStep",
    "ActionType",
]
