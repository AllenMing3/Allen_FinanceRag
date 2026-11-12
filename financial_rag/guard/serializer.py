"""
guard/serializer.py — 防幻觉结果序列化

将 ReliabilityService.check() / HallucinationGuard.check() 的原始结果序列化为 API 友好的结构。
所有调用点经 guard/service.py 统一入口收敛，本模块保证输出格式唯一。

设计原则:
- 每层保留 score + 诊断详情（而非仅 score）
- 不泄漏 raw 字段（内部原始输出）
- 向后兼容：前端已依赖的 overall_score / risk / passed / layers 结构不变
- skipped 输入（空 answer / 空 sources）→ 透出 skipped 标记，前端显示「未校验」
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
                "L4_structure_compliance": {"score": 0.75, "missing_sections": ["风险提示"]},
                ...
            },
        }
    """
    # 显式 skipped：未校验不是 0 分，不能伪装成「校验过且很差」
    if guard_result.get("skipped"):
        return {
            "skipped": True,
            "skip_reason": guard_result.get("skip_reason", ""),
            "overall_score": 0.0,
            "risk": "unknown",
            "passed": False,
            "layers": {},
        }

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
        "skipped": False,
        "skip_reason": "",
        "overall_score": round(guard_result.get("overall_score", 0), 3),
        "risk": guard_result.get("risk", "unknown"),
        "passed": guard_result.get("passed", False),
        "layers": layers,
    }
