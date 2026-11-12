"""
Test L3 LLM-as-judge 离线部分 — RAGAS faithfulness 评估器（eval/ragas_judge.py）。

对应生产代码：financial_rag/eval/ragas_judge.py
链路定位：替代旧 guard L5/L6 手写 prompt LLM 审计层（已淘汰删除），
深度语义评判前移到离线 LLM-as-judge；运行时只保留 L1-L4 规则层
（见 test_grounded_generation_step7.py）。

只测离线纯函数部分（golden→RAGAS 样本映射 / 区分度校准 / 契约字段）；
真正触网的 run_faithfulness 需要 ragas 依赖 + DASHSCOPE_API_KEY，
不进 CI —— 通过 CLI 手动执行：python -m financial_rag.eval.ragas_judge
"""
import pytest

from financial_rag.eval.ragas_judge import (
    DEFAULT_GAP_THRESHOLD,
    build_metric_inputs,
    build_ragas_samples,
    calibrate,
    extract_brief_claims,
    format_calibration,
    load_cases,
)


# ===================== golden → RAGAS 样本映射 =====================

class TestSampleMapping:
    def test_case_count_matches_golden(self):
        """样本数必须与 output_runner 同一份 golden 完全一致（不漏不重）。"""
        cases = load_cases()
        samples = build_ragas_samples(cases)
        assert len(samples) == len(cases)
        assert [s["case_id"] for s in samples] == [c["case_id"] for c in cases]

    def test_sample_shape_faithfulness_ready(self):
        """faithfulness 只需 answer+contexts：字段齐全、contexts 为非空文本列表。"""
        samples = build_ragas_samples(load_cases())
        for s in samples:
            assert s["answer"].strip(), f"{s['case_id']} answer 为空"
            assert isinstance(s["contexts"], list) and s["contexts"], (
                f"{s['case_id']} contexts 为空 —— faithfulness 无法评判"
            )
            assert all(isinstance(c, str) and c.strip() for c in s["contexts"])

    def test_template_scaffolding_stripped_from_answer(self):
        """分类元数据必须丢弃（与 L2 尺子同源）—— 否则 judge 把元数据行当无据声明扣分。"""
        samples = build_ragas_samples(load_cases())
        for s in samples:
            assert "**时间**" not in s["answer"] and "**影响**" not in s["answer"], (
                f"{s['case_id']} 模板标签未剥离: {s['answer'][:60]}"
            )
            assert "|" not in s["answer"], f"{s['case_id']} 分隔符未剥离"
        # 声明内容保留（标题正文不被误剥）
        revenue = next(s for s in samples if s["case_id"] == "nb_pos_sensetime_revenue_01")
        assert "50.3亿" in revenue["answer"]

    def test_only_claim_fields_kept_metadata_dropped(self):
        """只保留声明型字段(title/市场/展望)，丢弃时间/影响/涉及/行业分类元数据。

        元数据碎片喂 judge 会被当无据声明，实测把 good 档压到 0.6、gap 贴阈值边缘。
        """
        samples = {s["case_id"]: s for s in build_ragas_samples(load_cases())}
        auto = samples["nb_pos_sensetime_auto_04"]["answer"]
        # 声明型字段保留
        assert "商汤绝影与超30家车企合作" in auto
        assert "覆盖L2+到L4级别方案" in auto
        assert "布局东南亚和中东市场" in auto
        # 分类元数据必须已丢弃（时间值/影响级别/主体/行业标签）
        for noise in ("2024年度", "中等", "自动驾驶\n", "商汤绝影\n"):
            assert noise not in auto, f"元数据未丢弃: {noise!r}"
        # 每条 good 声明都能单独成行（3 行）
        assert len([ln for ln in auto.split("\n") if ln.strip()]) == 3

    def test_extract_brief_claims_defect_field_kept(self):
        """缺陷声明必须落在保留字段里（编造数字在 title，无据预测在展望）。"""
        cases = {c["case_id"]: c for c in load_cases()}
        fabnum = extract_brief_claims(cases["nb_bad_fabnum_01"]["brief_output"])
        assert "88亿" in fabnum, "编造数字必须在提取后的声明里，否则 judge 无法抓"
        ungrounded = extract_brief_claims(cases["nb_bad_ungrounded_01"]["brief_output"])
        assert "500港元" in ungrounded, "无据预测必须在提取后的声明里"

    def test_extract_brief_claims_no_claim_raises(self):
        """边界：非 NEWS_BRIEF 模板 / 全占位符 → 无声明可提，显式报错而非喂空文本。"""
        with pytest.raises(ValueError, match="声明型字段"):
            extract_brief_claims("   ")
        with pytest.raises(ValueError, match="声明型字段"):
            extract_brief_claims("一段没有模板结构的普通文本")

    def test_multi_template_real_claims_not_headings(self):
        """P0-1：quick_qa/deep_analysis 必须提取真实声明，不得只喂标题。

        旧实现标题正则把 '### 回答' 截成 '# 回答'、deep 只拿 '深度分析报告'，
        judge 评的是空标题 —— 静默正确性漏洞，与 L2 尺子同源修复。
        """
        samples = {s["case_id"]: s for s in build_ragas_samples(load_cases())}
        qq = samples["qq_pos_zhipu_funding_01"]["answer"]
        assert "融资金额超50亿元" in qq, f"quick_qa 真实声明丢失: {qq!r}"
        assert "#" not in qq, f"quick_qa 标题脚手架未剥离: {qq!r}"
        da = samples["da_pos_cambricon_01"]["answer"]
        assert "11.74亿" in da and "4.43亿" in da, f"deep 段落声明丢失: {da!r}"
        assert "深度分析报告" not in da, f"deep 报告标题未剥离: {da!r}"

    def test_contexts_come_from_source_docs(self):
        """contexts 必须取自 golden source_docs 的 text（与 Guard l1/l2 的 sources 同源）。"""
        cases = load_cases()
        samples = build_ragas_samples(cases)
        for c, s in zip(cases, samples):
            expected = [d["text"] for d in c["source_docs"] if d.get("text")]
            assert s["contexts"] == expected

    def test_tier_and_tags_carried_for_calibration(self):
        """quality_tier/tags 必须随样本携带 —— 校准分层依赖它们。"""
        samples = build_ragas_samples(load_cases())
        tiers = {s["quality_tier"] for s in samples}
        assert tiers == {"good", "bad"}
        assert any("fabricated_number" in s["tags"] for s in samples)

    def test_empty_answer_or_contexts_raises(self):
        """边界加固：空 answer / 空 contexts 必须显式报错，不得喂给 RAGAS 产生 NaN。"""
        good_case = load_cases()[0]
        empty_answer = {**good_case, "brief_output": "   "}
        with pytest.raises(ValueError):
            build_ragas_samples([empty_answer])
        empty_ctx = {**good_case, "source_docs": [{"doc_id": "d1", "text": ""}]}
        with pytest.raises(ValueError, match="非空 contexts"):
            build_ragas_samples([empty_ctx])

    def test_metric_inputs_match_v2_ascore_kwargs(self):
        """v2 接口约定键名：Faithfulness.ascore(user_input, response, retrieved_contexts)。"""
        samples = build_ragas_samples(load_cases())
        inputs = build_metric_inputs(samples)
        assert len(inputs) == len(samples)
        for s, inp in zip(samples, inputs):
            assert set(inp) == {"user_input", "response", "retrieved_contexts"}
            assert inp["response"] == s["answer"]
            assert inp["retrieved_contexts"] == s["contexts"]
            assert inp["user_input"] == s["question"]


# ===================== 区分度校准（合成得分，离线确定） =====================

def _synthetic_scores(samples, good=0.95, bad=0.40, fabnum=0.20):
    """按 tier/tag 构造合成 faithfulness 得分（与 samples 同序）。"""
    return [
        fabnum if "fabricated_number" in s["tags"]
        else (good if s["quality_tier"] == "good" else bad)
        for s in samples
    ]


class TestCalibration:
    def test_well_separated_scores_pass(self):
        samples = build_ragas_samples(load_cases())
        calib = calibrate(samples, _synthetic_scores(samples))
        assert calib["gap_ok"] is True and calib["fabnum_ok"] is True
        assert calib["violations"] == []
        # gap = good − 缺陷子集(fabnum@0.20 + ungrounded@0.40)，missfield 正交不参与。
        # 子集大小按 tag 动态统计，不硬编码 golden 构成（P0-1 多源扩容后自动适配）。
        n_fab = sum("fabricated_number" in s["tags"] for s in samples)
        n_ung = sum("ungrounded_claim" in s["tags"] for s in samples)
        assert n_fab > 0 and n_ung > 0, "缺陷子集不得为空"
        expected_gap = 0.95 - (0.20 * n_fab + 0.40 * n_ung) / (n_fab + n_ung)
        assert calib["gap"] == pytest.approx(expected_gap, abs=1e-4)
        assert calib["missfield_mean"] == pytest.approx(0.40, abs=1e-4)

    def test_no_separation_violates(self):
        """good/缺陷无区分度 → 必须报 violation（judge 失去分辨力即校准失败）。"""
        samples = build_ragas_samples(load_cases())
        calib = calibrate(samples, [0.9] * len(samples))
        assert calib["gap_ok"] is False
        assert any("方向性不足" in v for v in calib["violations"])

    def test_defect_not_detected_violates(self):
        """缺陷检出率：缺陷 case 得满分(1.0) → judge 未拆出任何无据声明，必须报 violation。"""
        samples = build_ragas_samples(load_cases())
        calib = calibrate(samples, [1.0] * len(samples))
        assert calib["detection_ok"] is False and calib["detection_rate"] < 1.0
        assert any("缺陷未检出" in v for v in calib["violations"])

    def test_fabnum_not_caught_violates(self):
        """编造数字子集得分不低于 good → 必须报 violation（RAGAS 应抓住编造数字）。"""
        samples = build_ragas_samples(load_cases())
        scores = _synthetic_scores(samples, good=0.8, bad=0.3, fabnum=0.85)
        calib = calibrate(samples, scores)
        assert calib["fabnum_ok"] is False
        assert any("编造数字未抓住" in v for v in calib["violations"])

    def test_per_case_aligns_with_samples(self):
        samples = build_ragas_samples(load_cases())
        scores = _synthetic_scores(samples)
        calib = calibrate(samples, scores)
        assert len(calib["per_case"]) == len(samples)
        assert [pc["case_id"] for pc in calib["per_case"]] == [s["case_id"] for s in samples]
        assert all(0.0 <= pc["faithfulness"] <= 1.0 for pc in calib["per_case"])

    def test_missing_tier_violates(self):
        """边界加固：faithfulness 缺陷子集缺失必须报 violation 而非静默通过。

        缺陷子集按 tag 选取（与 tier 正交），故清空缺陷标签模拟分层缺失。
        """
        samples = build_ragas_samples(load_cases())
        no_defects = [{**s, "tags": [t for t in s["tags"]
                                     if t not in ("fabricated_number", "ungrounded_claim")]}
                      for s in samples]
        calib = calibrate(no_defects, [0.9] * len(no_defects))
        assert any("分层缺失" in v for v in calib["violations"])

    def test_report_renders_pass_and_fail(self):
        samples = build_ragas_samples(load_cases())
        ok_report = format_calibration(calibrate(samples, _synthetic_scores(samples)))
        assert "PASS" in ok_report and "校准通过" in ok_report
        bad_report = format_calibration(calibrate(samples, [0.9] * len(samples)))
        assert "FAIL" in bad_report and "校准告警" in bad_report


# ===================== 契约钉死 =====================

class TestContracts:
    def test_default_gap_threshold_locked(self):
        """默认区分度参考线钉死 0.15 —— 仅报告用（辅定位，不阻断），调整需同步本用例。"""
        assert DEFAULT_GAP_THRESHOLD == 0.15

    def test_main_auxiliary_exit_zero_on_weak_gap(self, monkeypatch, tmp_path, capsys):
        """辅助定位契约：跑通产出得分即 exit 0，即使区分度未达参考线（弱 gap）。

        主门禁在 L1/L2 尺子 + step7；RAGAS 低分不应当作硬失败。monkeypatch 掉触网的
        run_faithfulness 与落盘路径，离线验证退出码与辅助观测提示。
        """
        import financial_rag.eval.ragas_judge as rj
        # 平得分 → gap=0，必然产生 violations（方向性不足）
        monkeypatch.setattr(rj, "run_faithfulness", lambda s, judge_model=None: [0.8] * len(s))
        monkeypatch.setattr(rj, "RESULTS_PATH", tmp_path / "ragas_results.json")
        rc = rj.main([])
        assert rc == 0, "弱 gap 不得导致非零退出（RAGAS 是辅助观测，非门禁）"
        out = capsys.readouterr().out
        assert "辅助观测" in out and "主门禁" in out
        # 得分仍然落盘（架子完整）
        assert (tmp_path / "ragas_results.json").exists()

    def test_structural_failure_exits_nonzero(self, monkeypatch, tmp_path):
        """禁止静默跳过：结构性故障（NaN / 缺 key）仍须 exit 2，与辅助定位不矛盾。"""
        import financial_rag.eval.ragas_judge as rj

        def _boom(samples, judge_model=None):
            raise RuntimeError("RAGAS 对 2 条样本返回 NaN 得分")
        monkeypatch.setattr(rj, "run_faithfulness", _boom)
        monkeypatch.setattr(rj, "RESULTS_PATH", tmp_path / "ragas_results.json")
        assert rj.main([]) == 2

    def test_runtime_guard_has_no_llm_dependency(self):
        """运行时 Guard 必须零 LLM 依赖（旧 L5/L6 已删，构造函数不收 llm）。

        深度语义评判只在离线 ragas_judge 通道 —— 若变红说明 LLM 层被误恢复回运行时。
        """
        import inspect

        from financial_rag.guard.reflector import HallucinationGuard
        sig = inspect.signature(HallucinationGuard.__init__)
        assert "llm" not in sig.parameters
        assert not hasattr(HallucinationGuard, "LLM_TRIGGER_THRESHOLD")

    def test_judge_llm_missing_key_raises_explicitly(self, monkeypatch):
        """禁止静默跳过：缺 DASHSCOPE_API_KEY 必须显式报错，不得静默返回。

        直接 patch os.getenv 模拟无 key 环境 —— get_judge_llm 内部会 load_dotenv
        且 find_dotenv 从模块路径向上搜索，chdir/delenv 都无法可靠隔离项目 .env。
        """
        import financial_rag.eval.ragas_judge as rj
        monkeypatch.setattr(rj.os, "getenv", lambda *a, **kw: "")
        with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
            rj.get_judge_llm()
