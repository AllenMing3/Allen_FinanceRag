"""
financial_rag.guard — 防幻觉与反思层

独立模块，与 core/ 平级，职责清晰：
- HallucinationGuard: 六层递进式防幻觉校验 (编排器)
- rule_layers: L1-L4 规则层实现
- llm_critique: L5 LLM质疑层
- llm_assist: L6 LLM协助层
- ReflectionLoop: ReAct 反思循环引擎

用法:
    from financial_rag.guard import HallucinationGuard
    from financial_rag.guard.rule_layers import l1_source_grounding
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
from .rule_layers import (
    l1_source_grounding,
    l2_numerical_fidelity,
    l3_citation_integrity,
    l4_structure_compliance,
)

__all__ = [
    "HallucinationGuard",
    "ReflectionLoop",
    "ReflectionConfig",
    "ReflectionState",
    "ThoughtStep",
    "ActionType",
    "l1_source_grounding",
    "l2_numerical_fidelity",
    "l3_citation_integrity",
    "l4_structure_compliance",
]
