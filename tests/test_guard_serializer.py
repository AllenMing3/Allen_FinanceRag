"""
tests/test_guard_serializer.py — 防幻觉诊断数据完整透出 + L5/L6 跳过信息同步

TDD: 先写测试，再改实现。
覆盖:
1. HallucinationGuard.check() — L5/L6 跳过时 checks 中有显式标记
2. serialize_guard_result() — 每层保留诊断详情而非仅 score
3. serialize_guard_result() — 跳过层包含 skipped + skip_reason
"""
import pytest

from financial_rag.guard.reflector import HallucinationGuard


# ===================== 测试数据 =====================

ANSWER_GROUNDED = (
    "商汤科技2024年实现营业收入50.3亿元，同比增长36.4%。"
    "生成式AI业务收入占比达60%，成为核心增长引擎。"
    "全年净亏损42.1亿元，同比收窄28.3%。"
)

SOURCES_GROUNDED = [
    {"text": "商汤科技2024年实现营业收入50.3亿元，同比增长36.4%。"
             "生成式AI业务收入占比达60%。全年净亏损42.1亿元，同比收窄28.3%。"}
]


# ===================== 1. L5/L6 跳过信息 =====================

class TestSkipInfo:
    """L5/L6 跳过时，checks 中必须有显式标记，不能静默省略"""

    def test_skipped_layers_present_in_checks(self):
        """无 LLM 时 L5/L6 应出现在 checks 中，标记为 skipped"""
        guard = HallucinationGuard(llm=None)
        result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        checks = result["checks"]

        # L5/L6 必须在 checks 中（不能静默省略）
        assert "L5_llm_critique" in checks, "L5 被静默省略，违反'不接受静默少测'"
        assert "L6_llm_assist" in checks, "L6 被静默省略，违反'不接受静默少测'"

    def test_skipped_layers_have_skip_flag(self):
        """跳过的层应有 skipped=True 和 skip_reason"""
        guard = HallucinationGuard(llm=None)
        result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        checks = result["checks"]

        for layer_key in ("L5_llm_critique", "L6_llm_assist"):
            layer = checks[layer_key]
            assert layer.get("skipped") is True, f"{layer_key} 缺少 skipped 标记"
            assert "skip_reason" in layer, f"{layer_key} 缺少 skip_reason"
            assert len(layer["skip_reason"]) > 0

    def test_skip_reason_no_llm(self):
        """无 LLM 时 skip_reason 应说明原因"""
        guard = HallucinationGuard(llm=None)
        result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        l5 = result["checks"]["L5_llm_critique"]
        assert "LLM" in l5["skip_reason"] or "llm" in l5["skip_reason"].lower()

    def test_skip_reason_rule_passed(self):
        """规则层达标时 skip_reason 应说明规则层已通过"""
        # 构造一个高 grounded 的 answer，让规则层分数 >= 0.85
        # 用 mock LLM 但规则层通过 → L5/L6 因规则达标而跳过
        guard = HallucinationGuard(llm=None)  # 无 LLM 也能测 skip_reason 逻辑
        result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        # 无 LLM 时 reason 是 "no LLM"，这里只验证字段存在
        assert "skip_reason" in result["checks"]["L5_llm_critique"]

    def test_executed_layers_no_skip_flag(self):
        """实际执行的层（L1-L4）不应有 skipped 标记"""
        guard = HallucinationGuard(llm=None)
        result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        for layer_key in ("L1_source_grounding", "L2_numerical_fidelity",
                          "L3_citation_integrity", "L4_structure_compliance"):
            layer = result["checks"][layer_key]
            assert layer.get("skipped") is not True, f"{layer_key} 不应标记为 skipped"

    def test_llm_layers_active_field(self):
        """llm_layers_active 字段应反映 L5/L6 是否真正执行"""
        guard = HallucinationGuard(llm=None)
        result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        assert result["llm_layers_active"] is False


# ===================== 2. serialize_guard_result =====================

class TestSerializer:
    """serialize_guard_result() 应保留完整诊断数据"""

    def test_import(self):
        from financial_rag.guard.serializer import serialize_guard_result
        assert callable(serialize_guard_result)

    def test_basic_structure(self):
        """返回值包含 overall_score / risk / passed / layers"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        assert "overall_score" in serialized
        assert "risk" in serialized
        assert "passed" in serialized
        assert "layers" in serialized

    def test_l1_diagnostic_details(self):
        """L1 层应包含 anchored/total/unanchored 诊断信息"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        l1 = serialized["layers"]["L1_source_grounding"]
        assert "score" in l1
        assert "anchored" in l1, "L1 缺少 anchored 诊断"
        assert "total" in l1, "L1 缺少 total 诊断"
        assert "unanchored" in l1, "L1 缺少 unanchored 列表"

    def test_l2_diagnostic_details(self):
        """L2 层应包含 verified/total/unmatched 诊断信息"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        l2 = serialized["layers"]["L2_numerical_fidelity"]
        assert "score" in l2
        assert "verified" in l2, "L2 缺少 verified 诊断"
        assert "unmatched" in l2, "L2 缺少 unmatched 列表"

    def test_l3_diagnostic_details(self):
        """L3 层应包含引用诊断"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        l3 = serialized["layers"]["L3_citation_integrity"]
        assert "score" in l3
        assert "citations_found" in l3

    def test_l4_diagnostic_details(self):
        """L4 层应包含 found_sections/missing_sections"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        l4 = serialized["layers"]["L4_structure_compliance"]
        assert "score" in l4
        assert "found_sections" in l4
        assert "missing_sections" in l4

    def test_skipped_layer_serialized(self):
        """跳过的层序列化后应有 skipped + skip_reason"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        l5 = serialized["layers"]["L5_llm_critique"]
        assert l5.get("skipped") is True
        assert "skip_reason" in l5

        l6 = serialized["layers"]["L6_llm_assist"]
        assert l6.get("skipped") is True
        assert "skip_reason" in l6

    def test_scores_are_rounded(self):
        """所有 score 应保留 3 位小数"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        for key, layer in serialized["layers"].items():
            if "score" in layer:
                score_str = str(layer["score"])
                if "." in score_str:
                    decimals = len(score_str.split(".")[1])
                    assert decimals <= 3, f"{key} score 精度超过 3 位: {layer['score']}"

    def test_no_raw_field_leaked(self):
        """序列化结果不应包含 raw 字段（LLM 原始输出）"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        for key, layer in serialized["layers"].items():
            assert "raw" not in layer, f"{key} 泄漏了 raw 字段"

    def test_overall_not_in_layers(self):
        """overall 不应出现在 layers 中"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        assert "overall" not in serialized["layers"]


# ===================== 3. 向后兼容 =====================

class TestBackwardCompat:
    """确保现有字段不丢失，前端不会 break"""

    def test_top_level_fields(self):
        """序列化结果保留前端已依赖的顶层字段"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        # 前端 analyze.js / query.js 依赖这些字段
        assert "overall_score" in serialized
        assert "risk" in serialized
        assert "passed" in serialized
        assert "layers" in serialized
        assert isinstance(serialized["layers"], dict)

    def test_layer_score_always_present(self):
        """每层必须有 score 字段（前端渲染依赖）"""
        from financial_rag.guard.serializer import serialize_guard_result
        guard = HallucinationGuard(llm=None)
        guard_result = guard.check(ANSWER_GROUNDED, SOURCES_GROUNDED)
        serialized = serialize_guard_result(guard_result)

        for key, layer in serialized["layers"].items():
            assert "score" in layer, f"{key} 缺少 score 字段"
