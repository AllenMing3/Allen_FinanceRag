"""
guard/serializer.py — 防幻觉结果序列化

将 HallucinationGuard.check() 的原始结果序列化为 API 友好的结构。
统一 3 个调用点（kb_router / analysis_router fallback / agent chain）的输出格式。

设计原则:
- 每层保留 score + 诊断详情（而非仅 score）
- 跳过的层包含 skipped=True + skip_reason（禁止静默省略）
- 不泄漏 raw 字段（LLM 原始输出）
- 向后兼容：前端已依赖的 overall_score / risk / passed / layers 结构不变
"""
from typing import Dict, Any


# 每层允许透出的诊断字段白名单（排除 raw、内部标记等）
_LAYER_FIELDS = {
    "L1_source_grounding": [
        "score", "passed", "anchored", "total", "unanchored",
    ],
    "L2_numerical_fidelity": [
        "score", "passed", "verified", "total", "unmatched",
    ],
    "L3_citation_integrity": [
        "score", "passed", "citations_found", "valid", "invalid",
        "has_text_reference", "mode",
    ],
    "L4_structure_compliance": [
        "score", "passed", "found_sections", "missing_sections", "mode",
    ],
    "L5_llm_critique": [
        "score", "passed", "false_positives", "false_negatives",
        "overreach", "critique_summary", "rule_blind_spot_count",
        # 跳过时的字段
        "skipped", "skip_reason",
    ],
    "L6_llm_assist": [
        "score", "passed", "total_claims", "supported_claims",
        "fabricated_count", "unsupported_claims", "fabricated_claims",
        "per_sentence", "cross_sentence_issues", "overall_assessment",
        # 跳过时的字段
        "skipped", "skip_reason",
    ],
}


def serialize_guard_result(guard_result: Dict[str, Any]) -> Dict[str, Any]:
    """将 HallucinationGuard.check() 结果序列化为 API 输出格式。

    Args:
        guard_result: HallucinationGuard.check() 的返回值

    Returns:
        {
            "overall_score": 0.85,
            "risk": "low",
            "passed": True,
            "layers": {
                "L1_source_grounding": {"score": 1.0, "anchored": 3, "total": 3, "unanchored": []},
                "L5_llm_critique": {"score": 0.0, "skipped": True, "skip_reason": "..."},
                ...
            },
        }
    """
    checks = guard_result.get("checks", {})

    layers = {}
    for key, val in checks.items():
        if key == "overall":
            continue
        if not isinstance(val, dict):
            continue

        allowed = _LAYER_FIELDS.get(key)
        if allowed:
            layer = {}
            for field in allowed:
                if field in val:
                    v = val[field]
                    # score 保留 3 位小数
                    if field == "score" and isinstance(v, (int, float)):
                        v = round(v, 3)
                    layer[field] = v
            layers[key] = layer
        else:
            # 未知层：只保留 score
            layers[key] = {"score": round(val.get("score", 0), 3)}

    return {
        "overall_score": round(guard_result.get("overall_score", 0), 3),
        "risk": guard_result.get("risk", "unknown"),
        "passed": guard_result.get("passed", False),
        "layers": layers,
    }
