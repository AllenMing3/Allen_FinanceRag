"""
L2 输出层质量指标单测 — 对称 test_eval_metrics.py。

钉死 NEWS_BRIEF 输出质量指标的确定性行为（探针已验证 Guard L1/L2 复用点）：
  - field_completeness: 7 字段有效填充率（新定制，占位符/空值判未填充）
  - numerical_fidelity: 复用 Guard l2，编造数字进 unmatched_numbers
  - source_grounding:   复用 Guard l1，无源声明进 unanchored
  - output_quality:     加权聚合，正例 > 负例（区分度）
全部离线确定、无需 LLM、无需检索器。
"""
import pytest

from financial_rag.eval.output_metrics import (
    OUTPUT_QUALITY_WEIGHTS,
    field_completeness,
    evaluate_case,
)

# 规范正例：数字带单位、source 完整 —— 探针验证 L1/L2 均为满分 1.0
GOOD_BRIEF = """## 📰 商汤2024年营收50.3亿元同比增36.4%
**时间**: 2024年度 | **影响**: 重大
**涉及**: 商汤科技
**行业**: 生成式AI,算力
**市场**: 研发投入45亿元占营收89.5%
**展望**: 力争三年内盈亏平衡"""

SOURCE = [{"doc_id": "d1", "text":
           "商汤科技发布2024年年度报告。全年实现营业收入50.3亿元，同比增长36.4%。"
           "生成式AI业务收入占比达60%。研发投入达45亿元，占营收比例89.5%。"
           "SenseCore大装置算力规模达4万卡。力争在未来三年内实现盈亏平衡。"}]


@pytest.mark.unit
class TestFieldCompleteness:
    """字段完整性 — 新定制指标（Guard L4 不匹配 NEWS_BRIEF 字段结构）。"""

    def test_all_fields_filled(self):
        r = field_completeness(GOOD_BRIEF)
        assert r["score"] == 1.0
        assert r["filled"] == 7 and r["total"] == 7
        assert r["missing_fields"] == []

    def test_placeholder_counts_as_missing(self):
        brief = GOOD_BRIEF.replace("力争三年内盈亏平衡", "（短期展望暂缺）")
        r = field_completeness(brief)
        assert "outlook" in r["missing_fields"]
        assert r["filled"] == 6
        assert r["score"] == pytest.approx(6 / 7)

    def test_empty_value_counts_as_missing(self):
        brief = GOOD_BRIEF.replace("**展望**: 力争三年内盈亏平衡", "**展望**:")
        r = field_completeness(brief)
        assert "outlook" in r["missing_fields"]

    def test_missing_title_detected(self):
        brief = GOOD_BRIEF.replace(
            "## 📰 商汤2024年营收50.3亿元同比增36.4%", "商汤营收增长")
        r = field_completeness(brief)
        assert "event_title" in r["missing_fields"]


@pytest.mark.unit
class TestNumericalFidelity:
    """数值忠实度 — 复用 Guard l2_numerical_fidelity，新闻简报最怕编造数字。"""

    def test_grounded_numbers_full_score(self):
        r = evaluate_case(GOOD_BRIEF, SOURCE)
        assert r["numerical_fidelity"] == 1.0
        assert r["unmatched_numbers"] == []

    def test_fabricated_number_detected(self):
        brief = GOOD_BRIEF.replace("50.3亿元", "88亿元")
        r = evaluate_case(brief, SOURCE)
        assert r["numerical_fidelity"] < 1.0
        assert any("88" in n for n in r["unmatched_numbers"]), \
            f"编造的 88亿 未被检出: {r['unmatched_numbers']}"


@pytest.mark.unit
class TestSourceGrounding:
    """来源锚定 — 复用 Guard l1_source_grounding，检出无源声明。"""

    def test_grounded_claims_full_score(self):
        r = evaluate_case(GOOD_BRIEF, SOURCE)
        assert r["source_grounding"] == 1.0

    def test_ungrounded_claim_detected(self):
        brief = GOOD_BRIEF.replace("力争三年内盈亏平衡", "预计明年股价涨至500港元")
        r = evaluate_case(brief, SOURCE)
        assert r["source_grounding"] < 1.0
        assert r["unanchored"], "无源声明未被检出"


@pytest.mark.unit
class TestOutputQuality:
    """综合分聚合 + 区分度。"""

    def test_weights_sum_to_one(self):
        assert sum(OUTPUT_QUALITY_WEIGHTS.values()) == pytest.approx(1.0)

    def test_good_beats_fabricated(self):
        good = evaluate_case(GOOD_BRIEF, SOURCE)
        bad = evaluate_case(GOOD_BRIEF.replace("50.3亿元", "88亿元"), SOURCE)
        assert good["output_quality"] > bad["output_quality"]

    def test_good_case_near_ceiling(self):
        r = evaluate_case(GOOD_BRIEF, SOURCE)
        assert r["output_quality"] == pytest.approx(1.0)


@pytest.mark.unit
class TestEvaluateCaseShape:
    def test_returns_all_metric_keys(self):
        r = evaluate_case(GOOD_BRIEF, SOURCE)
        for key in ("numerical_fidelity", "source_grounding",
                    "field_completeness", "output_quality",
                    "unmatched_numbers", "missing_fields", "unanchored"):
            assert key in r, f"evaluate_case 缺返回键: {key}"
