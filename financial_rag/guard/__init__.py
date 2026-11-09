"""
financial_rag.guard — 防幻觉层

独立模块，与 core/ 平级，职责清晰：
- HallucinationGuard: L1-L4 规则层防幻觉校验 (编排器，确定性、零 LLM 依赖)
- rule_layers: L1-L4 规则层实现

深度语义评判（LLM-as-judge）已迁至离线 eval: financial_rag/eval/ragas_judge.py

用法:
    from financial_rag.guard import HallucinationGuard
    from financial_rag.guard.rule_layers import l1_source_grounding
"""
from .reflector import HallucinationGuard
from .rule_layers import (
    l1_source_grounding,
    l2_numerical_fidelity,
    l3_citation_integrity,
    l4_structure_compliance,
)

__all__ = [
    "HallucinationGuard",
    "l1_source_grounding",
    "l2_numerical_fidelity",
    "l3_citation_integrity",
    "l4_structure_compliance",
]
