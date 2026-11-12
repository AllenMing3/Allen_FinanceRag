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

    @pytest.mark.regression
    def test_template_labels_not_counted_as_claims(self):
        """模板脚手架（**时间**: 等标签词）不是声明，不得被 L1 计入 unanchored。

        Guard L1 阈值收紧到 0.30 后曾误判标签行（overlap 0.2）—— evaluate_case
        现在喂 Guard 前剥离脚手架，退回直传 brief_output 的旧实现即红。
        """
        r = evaluate_case(GOOD_BRIEF, SOURCE)
        assert all("**时间**" not in u for u in r["unanchored"])
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


# ===================== 多模板支持（P0-1：尺子不再只适配 NEWS_BRIEF） =====================

GOOD_QA = """### 回答
智谱AI完成B+轮融资，融资金额超50亿元[1]。

**数据支撑**: 融资金额超50亿元，投后估值超200亿元
**来源**: 来源：融资快讯
**可信度**: 高，官方宣布
"""

QA_SOURCE = [{"doc_id": "d_zhipu", "text":
              "智谱AI宣布完成B+轮融资，融资金额超50亿元，投后估值超200亿元。"
              "本轮融资由社保基金、北京市政府引导基金等联合投资。"}]

GOOD_DEEP = """## 深度分析报告

### 摘要
寒武纪2024年营收11.74亿元，同比增长65.56%，净亏损同比收窄。

### 盈利能力
公司尚未盈利，全年净亏损4.43亿元，同比收窄47.8%。

### 成长性
营业收入11.74亿元，同比增长65.56%，云端产品线为收入主力。

### 财务健康度
经营活动现金流净额为-2.15亿元，存货账面价值3.87亿元。

### 估值参考
研发投入10.72亿元，占营收比例91.3%，估值需关注盈利拐点。

### 投资建议
持有：云端芯片放量但尚未盈利，关注思元690发布节奏。
"""

DEEP_SOURCE = [{"doc_id": "d_cambricon", "text":
                "寒武纪科技股份有限公司2024年年度报告。2024年度，寒武纪实现营业收入11.74亿元，"
                "同比增长65.56%。其中，云端产品线收入11.66亿元，占总营收99.3%。"
                "全年净亏损4.43亿元，同比收窄47.8%。研发投入10.72亿元，占营收比例91.3%。"
                "公司尚未实现盈利，经营活动现金流净额为-2.15亿元。存货账面价值3.87亿元。"
                "新一代思元690已完成流片，预计2025年Q2发布。"}]


@pytest.mark.unit
class TestTemplateSpecs:
    def test_quick_qa_all_fields_filled(self):
        r = field_completeness(GOOD_QA, template="quick_qa")
        assert r["score"] == 1.0
        assert r["total"] == 4
        assert r["missing_fields"] == []

    def test_quick_qa_placeholder_counts_as_missing(self):
        qa = GOOD_QA.replace("融资金额超50亿元，投后估值超200亿元", "（支撑数据暂缺）", 1)
        # 只替换数据支撑字段值（回答行不含该串，首个匹配在**数据支撑**行）
        r = field_completeness(qa, template="quick_qa")
        assert "supporting_data" in r["missing_fields"]
        assert r["score"] == pytest.approx(3 / 4)

    def test_deep_analysis_all_sections_filled(self):
        r = field_completeness(GOOD_DEEP, template="deep_analysis")
        assert r["score"] == 1.0
        assert r["total"] == 6
        assert r["missing_fields"] == []

    def test_deep_analysis_missing_section_detected(self):
        deep = GOOD_DEEP.replace("### 估值参考\n研发投入10.72亿元，占营收比例91.3%，估值需关注盈利拐点。\n\n", "")
        r = field_completeness(deep, template="deep_analysis")
        assert "valuation_hint" in r["missing_fields"]
        assert r["score"] == pytest.approx(5 / 6)

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="未知模板"):
            evaluate_case(GOOD_BRIEF, SOURCE, template="no_such_template")

    def test_default_template_backward_compatible(self):
        assert evaluate_case(GOOD_BRIEF, SOURCE) == evaluate_case(
            GOOD_BRIEF, SOURCE, template="news_brief")

    def test_quick_qa_scaffolding_stripped(self):
        """quick_qa 模板标签（**数据支撑** 等）不是声明，不得计入 unanchored。"""
        r = evaluate_case(GOOD_QA, QA_SOURCE, template="quick_qa")
        assert all("数据支撑" not in u for u in r["unanchored"])
        assert r["source_grounding"] == 1.0
        assert r["output_quality"] == pytest.approx(1.0)


# ===================== per-case 关键用例锁（P0-2：单 case 掉分不再被均值冲淡） =====================

from financial_rag.eval.output_runner import (
    OutputReport,
    compare,
    format_comparison,
    load_baseline,
    save_baseline,
)


def _mk_report(per_case):
    """构造最小 OutputReport：macro/gap 与 baseline 对齐，只变 per_case。"""
    return OutputReport(
        case_count=len(per_case), good_count=len(per_case), bad_count=0,
        macro={m: 1.0 for m in ("numerical_fidelity", "source_grounding",
                                 "field_completeness", "output_quality")},
        per_tier={}, per_tag={}, separation_gap=0.1, per_case=per_case,
    )


def _mk_baseline(per_case, critical=None):
    b = {"macro": {m: 1.0 for m in ("numerical_fidelity", "source_grounding",
                                     "field_completeness", "output_quality")},
         "separation_gap": 0.1}
    if critical is not None:
        b["critical_cases"] = critical
    return b


@pytest.mark.unit
class TestCriticalCaseLock:
    def test_single_case_drop_alerts(self):
        """单 case 0.9→0.3 被 macro 均值冲淡的场景：pin 下界必须单独报警。"""
        report = _mk_report({"c1": {"output_quality": 0.3}})
        cmp = compare(report, _mk_baseline({"c1": {"output_quality": 1.0}},
                                           critical={"c1": 0.95}))
        assert not cmp.ok
        hits = [r for r in cmp.regressions if r["scope"] == "case"]
        assert len(hits) == 1 and hits[0]["metric"] == "c1"
        assert hits[0]["baseline"] == 0.95 and hits[0]["current"] == 0.3
        assert "c1" in format_comparison(cmp)

    def test_critical_case_missing_alerts(self):
        """pin 的关键 case 从 golden 消失 → 回归告警，不得静默通过。"""
        report = _mk_report({"c2": {"output_quality": 1.0}})
        cmp = compare(report, _mk_baseline({}, critical={"c1": 0.95}))
        assert not cmp.ok
        hits = [r for r in cmp.regressions if r["scope"] == "case"]
        assert len(hits) == 1 and hits[0]["metric"] == "c1"
        assert "消失" in format_comparison(cmp)

    def test_update_baseline_preserves_pins(self, tmp_path):
        """--update-baseline 重生成不得覆盖手工 curated 的 critical_cases。"""
        import json
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(_mk_baseline({}, critical={"c1": 0.95})),
                        encoding="utf-8")
        save_baseline(_mk_report({"c1": {"output_quality": 1.0}}), path)
        saved = load_baseline(path)
        assert saved["critical_cases"] == {"c1": 0.95}
        assert saved["case_count"] == 1  # macro/counts 确实被重生成了

    def test_critical_case_met_no_false_alarm(self):
        """达标（≥下界）不误报；无 critical_cases 段的旧 baseline 行为不变。"""
        report = _mk_report({"c1": {"output_quality": 0.96}})
        cmp = compare(report, _mk_baseline({}, critical={"c1": 0.95}))
        assert cmp.ok and cmp.regressions == []
        cmp2 = compare(report, _mk_baseline({}))
        assert cmp2.ok
