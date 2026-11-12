"""
L2 输出测评 golden 自洽 + 区分度锁 — 对称 test_eval_golden.py。

钉死 output_news_brief.jsonl 的数据可靠性与评估器区分能力，全部离线确定、无需 LLM：
  1. schema 完整：case_id/query/source_docs/brief_output/expected 齐全，case_id 唯一
  2. 覆盖充分：正负例都有，三类缺陷（fabricated_number/missing_field/ungrounded_claim）全覆盖
  3. 正例零缺陷：unmatched_numbers/missing_fields 为空，output_quality 达下界
  4. 负例缺陷精确检出：按 defect_type 断言对应指标掉分、缺陷字段命中
  5. expected 回归锁：golden 记录的 unmatched/missing 与 evaluate_case 实际一致
  6. 区分度锁：正例均分 − 负例均分 ≥ 阈值，且正负例 output_quality 区间不重叠

对应数据：financial_rag/eval/golden/output_news_brief.jsonl
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from financial_rag.eval.output_metrics import evaluate_case

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "financial_rag" / "eval" / "golden" / "output_news_brief.jsonl"
)
_REQUIRED = {"case_id", "query", "source_docs", "brief_output", "expected"}
_EXP_REQUIRED = {
    "quality_tier", "min_output_quality", "max_output_quality",
    "unmatched_numbers", "missing_fields", "tags",
}
_DEFECT_TYPES = {"fabricated_number", "missing_field", "ungrounded_claim"}
# 正例均分 − 负例均分下界（实测 gap≈0.099，留 margin 防 golden 微调波动）
_SEPARATION_THRESHOLD = 0.05


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def cases():
    return _load_jsonl(_GOLDEN)


@pytest.fixture(scope="module")
def scored(cases):
    """[(case, evaluate_case 结果)] — module 级缓存，只跑一遍。

    template 透传：多模板 case 必须用对应尺子评估，否则 quick_qa/deep_analysis
    会被 news_brief 字段规格误判全缺（P0-1 修复点）。
    """
    return [(c, evaluate_case(c["brief_output"], c["source_docs"],
                              template=c.get("template", "news_brief")))
            for c in cases]


def _by_tier(cases, tier):
    return [c for c in cases if c["expected"]["quality_tier"] == tier]


@pytest.mark.unit
class TestGoldenSchema:
    def test_case_ids_unique(self, cases):
        ids = [c["case_id"] for c in cases]
        dup = [i for i, n in Counter(ids).items() if n > 1]
        assert len(ids) == len(set(ids)), f"case_id 重复: {dup}"

    def test_required_fields_present(self, cases):
        for c in cases:
            missing = _REQUIRED - set(c.keys())
            assert not missing, f"{c.get('case_id')} 缺字段: {missing}"

    def test_expected_fields_present(self, cases):
        for c in cases:
            missing = _EXP_REQUIRED - set(c["expected"].keys())
            assert not missing, f"{c['case_id']} expected 缺字段: {missing}"

    def test_quality_tier_valid(self, cases):
        for c in cases:
            assert c["expected"]["quality_tier"] in ("good", "bad"), \
                f"{c['case_id']} quality_tier 非法"

    def test_source_docs_non_empty(self, cases):
        for c in cases:
            assert c["source_docs"], f"{c['case_id']} source_docs 为空"
            for d in c["source_docs"]:
                assert d.get("doc_id") and d.get("text"), \
                    f"{c['case_id']} source_doc 缺 doc_id/text"

    def test_tags_non_empty(self, cases):
        for c in cases:
            assert c["expected"]["tags"], f"{c['case_id']} tags 为空"


@pytest.mark.unit
class TestCoverage:
    def test_has_positives(self, cases):
        assert _by_tier(cases, "good"), "golden 没有正例"

    def test_has_negatives(self, cases):
        assert _by_tier(cases, "bad"), "golden 没有负例"

    def test_all_defect_types_covered(self, cases):
        covered = {c["expected"].get("defect_type") for c in _by_tier(cases, "bad")}
        assert _DEFECT_TYPES <= covered, \
            f"三类缺陷未全覆盖，缺: {_DEFECT_TYPES - covered}"

    def test_multi_template_covered(self, cases):
        """P0-1：尺子不得只适配单模板 —— 三种渲染模板都要有样本。"""
        covered = {c.get("template", "news_brief") for c in cases}
        required = {"news_brief", "quick_qa", "deep_analysis"}
        assert required <= covered, f"模板覆盖不全，缺: {required - covered}"

    def test_multi_source_covered(self, cases):
        """P0-1：不得全部 case 共用单一源文档。"""
        doc_ids = {d["doc_id"] for c in cases for d in c["source_docs"]}
        assert len(doc_ids) >= 3, f"源文档覆盖不足: {doc_ids}"


@pytest.mark.unit
class TestPositives:
    def test_positive_no_defects(self, cases):
        for c in _by_tier(cases, "good"):
            exp = c["expected"]
            assert exp["unmatched_numbers"] == [], f"{c['case_id']} 正例不应有编造数字"
            assert exp["missing_fields"] == [], f"{c['case_id']} 正例不应缺字段"

    def test_positive_meets_min_quality(self, scored):
        for c, r in scored:
            if c["expected"]["quality_tier"] == "good":
                assert r["output_quality"] >= c["expected"]["min_output_quality"], \
                    f"{c['case_id']} 正例 oq={r['output_quality']:.3f} 低于下界"


@pytest.mark.unit
class TestNegatives:
    def test_defect_type_matches_tag(self, cases):
        for c in _by_tier(cases, "bad"):
            dt = c["expected"]["defect_type"]
            assert dt in c["expected"]["tags"], \
                f"{c['case_id']} defect_type={dt} 未出现在 tags"

    def test_negative_below_max_quality(self, scored):
        for c, r in scored:
            if c["expected"]["quality_tier"] == "bad":
                assert r["output_quality"] <= c["expected"]["max_output_quality"], \
                    f"{c['case_id']} 负例 oq={r['output_quality']:.3f} 高于上界"

    def test_fabricated_number_detected(self, scored):
        hit = 0
        for c, r in scored:
            if c["expected"].get("defect_type") == "fabricated_number":
                assert r["unmatched_numbers"], f"{c['case_id']} 编造数字未检出"
                assert r["numerical_fidelity"] < 1.0, f"{c['case_id']} 数值忠实度未掉分"
                hit += 1
        assert hit > 0, "没有 fabricated_number 负例被验证"

    def test_missing_field_detected(self, scored):
        hit = 0
        for c, r in scored:
            if c["expected"].get("defect_type") == "missing_field":
                assert r["missing_fields"], f"{c['case_id']} 缺失字段未检出"
                assert r["field_completeness"] < 1.0, f"{c['case_id']} 字段完整性未掉分"
                hit += 1
        assert hit > 0, "没有 missing_field 负例被验证"

    def test_ungrounded_claim_detected(self, scored):
        hit = 0
        for c, r in scored:
            if c["expected"].get("defect_type") == "ungrounded_claim":
                assert r["unanchored"], f"{c['case_id']} 无源句未检出"
                assert r["source_grounding"] < 1.0, f"{c['case_id']} 来源锚定未掉分"
                hit += 1
        assert hit > 0, "没有 ungrounded_claim 负例被验证"


@pytest.mark.unit
class TestExpectedRegression:
    """expected 回归锁 — golden 记录的缺陷必须与 evaluate_case 实际一致。"""

    def test_unmatched_numbers_match(self, scored):
        for c, r in scored:
            assert r["unmatched_numbers"] == c["expected"]["unmatched_numbers"], \
                f"{c['case_id']} unmatched 漂移: 实际{r['unmatched_numbers']} ≠ 期望{c['expected']['unmatched_numbers']}"

    def test_missing_fields_match(self, scored):
        for c, r in scored:
            assert r["missing_fields"] == c["expected"]["missing_fields"], \
                f"{c['case_id']} missing_fields 漂移: 实际{r['missing_fields']} ≠ 期望{c['expected']['missing_fields']}"


@pytest.mark.unit
class TestSeparation:
    """区分度锁 — 评估器必须能分开好坏样本（plan 验证标准核心）。"""

    def test_positive_mean_beats_negative_mean(self, scored):
        good = [r["output_quality"] for c, r in scored if c["expected"]["quality_tier"] == "good"]
        bad = [r["output_quality"] for c, r in scored if c["expected"]["quality_tier"] == "bad"]
        gap = sum(good) / len(good) - sum(bad) / len(bad)
        assert gap >= _SEPARATION_THRESHOLD, \
            f"区分度不足: 正例均分-负例均分={gap:.4f} < {_SEPARATION_THRESHOLD}"

    def test_no_score_overlap(self, scored):
        """每个正例 output_quality 严格高于每个负例 —— 无重叠才可二分类。"""
        good = [r["output_quality"] for c, r in scored if c["expected"]["quality_tier"] == "good"]
        bad = [r["output_quality"] for c, r in scored if c["expected"]["quality_tier"] == "bad"]
        assert min(good) > max(bad), \
            f"正负例分数重叠: min(good)={min(good):.3f} <= max(bad)={max(bad):.3f}"
