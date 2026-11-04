"""
L1 检索层评估回归锁 — 把 runner.evaluate() 的数字变成 CI 护栏。

对应 plan E（接线）+ Test Plan #3/#5，全部 @pytest.mark.eval：
  1. 离线确定性：evaluate() 连跑两次 to_dict() 完全一致（无网络、无随机）
  2. fusion 回归锁：fusion 通道 recall/mrr/ndcg >= baseline - tol，回退即失败
  3. bug_scene 达标：今天修的 bug 场景子集（未登录词 / 多 chunk 长文）不低于 baseline，
     并有绝对底线（recall >= 0.5）证明修复确实生效，而非仅"没变差"
  4. 分通道可见性：bm25/vector/fusion 三通道独立非零、指标合法，不静默坍塌
     （Step3 "Jaccard 融合淹没 BM25 改善" 教训的制度化护栏）
  5. Step 5 过滤评估：零泄漏（返回项不违反过滤条件）+ 零 trap 命中 +
     filter-aware 召回达标（过滤器不得误杀相关文档）
  6. Step 5 门控评估：库内查询不误拦、分数门控（含阈值边界）决策正确、
     拦截诊断字段完整；零重叠库外查询拦不住为已知局限（棘轮断言，修复即纳入）

快跑通道用 `-m "not eval"` 排除；显式跑：pytest tests/test_eval_retrieval.py -m eval
baseline.json 由 `python -m financial_rag.eval.runner --update-baseline` 生成并纳入 git。
"""
import pytest

from financial_rag.eval.metrics import METRIC_NAMES
from financial_rag.eval.runner import (
    CHANNELS, DEFAULT_TOP_K, DEFAULT_TOL, GUARDED_METRICS,
    evaluate, load_baseline, load_filter_cases, load_gate_cases,
)

# bug_scene 绝对达标底线：至少召回一半相关文档，证明修复生效（非仅"未回退"）
_BUG_SCENE_RECALL_FLOOR = 0.5


@pytest.fixture(scope="module")
def report():
    """模块内共享一次 evaluate()（~0.7s），避免每个用例重复装配检索器。"""
    return evaluate(top_k=DEFAULT_TOP_K)


@pytest.fixture(scope="module")
def baseline():
    b = load_baseline()
    assert b is not None, (
        "baseline.json 不存在，无法做回归对比。先跑: "
        "python -m financial_rag.eval.runner --update-baseline"
    )
    return b


@pytest.mark.eval
class TestOfflineDeterminism:
    """Test Plan #3：同 golden 连跑两次，各指标完全一致（无网络、无随机）。"""

    def test_evaluate_twice_identical(self):
        r1 = evaluate(top_k=DEFAULT_TOP_K)
        r2 = evaluate(top_k=DEFAULT_TOP_K)
        assert r1.to_dict() == r2.to_dict(), "evaluate() 非确定：两次运行结果不一致"

    def test_long_doc_triggers_chunking(self, report):
        # 长文 d_sensetime_long 经 chunker 切分 → chunk 数应 > 文档数
        assert report.chunk_count > report.doc_count, (
            f"chunk_count({report.chunk_count}) 未超过 doc_count({report.doc_count})，"
            "长文未触发切分，无法覆盖 BM25 doc_key 碰撞场景"
        )


@pytest.mark.eval
class TestFusionNoRegression:
    """plan E 回归锁：fusion 通道核心指标不得低于 baseline - tol。"""

    def test_fusion_core_metrics_at_or_above_baseline(self, report, baseline):
        base_macro = baseline["channels"]["fusion"]["macro"]
        for m in GUARDED_METRICS:
            cur = report.macro("fusion", m)
            floor = base_macro[m] - DEFAULT_TOL
            assert cur >= floor, (
                f"fusion.{m} 回归：当前 {cur:.4f} < baseline {base_macro[m]:.4f} "
                f"- tol {DEFAULT_TOL} = {floor:.4f}"
            )

    def test_baseline_metadata_matches(self, report, baseline):
        # baseline 须与当前 golden 规模一致，防止拿旧基线锁新数据（错配即失效）
        assert baseline["query_count"] == report.query_count, (
            f"baseline query_count={baseline['query_count']} 与当前 {report.query_count} 不符，"
            "golden 数据已变但基线未更新，请重跑 --update-baseline"
        )
        assert baseline["doc_count"] == report.doc_count
        assert baseline["top_k"] == report.top_k


@pytest.mark.eval
class TestBugSceneSubset:
    """plan E：tags=bug_scene 子集单独断言达标（今天修的 bug 场景不许回退）。"""

    def test_bug_scene_fusion_meets_baseline(self, report, baseline):
        base_bug = baseline["channels"]["fusion"]["per_tag"]["bug_scene"]
        cur_bug = report.channels["fusion"].per_tag["bug_scene"]
        for m in GUARDED_METRICS:
            floor = base_bug[m] - DEFAULT_TOL
            assert cur_bug[m] >= floor, (
                f"bug_scene fusion.{m} 回归：当前 {cur_bug[m]:.4f} < "
                f"baseline {base_bug[m]:.4f} - tol {DEFAULT_TOL} = {floor:.4f}"
            )

    def test_bug_scene_recall_above_absolute_floor(self, report):
        # 绝对底线：bug 场景至少召回一半相关文档，证明修复真的生效
        recall = report.channels["fusion"].per_tag["bug_scene"]["recall"]
        assert recall >= _BUG_SCENE_RECALL_FLOOR, (
            f"bug_scene fusion recall={recall:.4f} < {_BUG_SCENE_RECALL_FLOOR}，"
            "未登录词 / 长文场景修复疑似失效"
        )


@pytest.mark.eval
class TestChannelVisibility:
    """Test Plan #5：三通道独立可见、不静默坍塌（Step3 融合淹没 BM25 教训）。"""

    def test_all_channels_reported(self, report):
        assert set(report.channels.keys()) == set(CHANNELS)

    def test_each_channel_retrieves_nonzero(self, report):
        # 每通道 recall 必须 > 0，证明该通道真的检索到东西（非静默空 —— Step1 教训）
        for ch in CHANNELS:
            assert report.macro(ch, "recall") > 0.0, f"{ch} 通道 recall=0，疑似静默失效"

    def test_all_metrics_in_unit_range(self, report):
        for ch in CHANNELS:
            for m in METRIC_NAMES:
                v = report.macro(ch, m)
                assert 0.0 <= v <= 1.0, f"{ch}.{m}={v:.4f} 超出 [0,1] 合法区间"


@pytest.mark.eval
class TestFilterEval:
    """Step 5 过滤评估回归锁（golden/filter_cases.jsonl）。

    核心契约：过滤器是硬约束 —— 泄漏（返回违反过滤条件的项）和
    trap 命中（应被过滤排除的主题强相关文档）都是零容忍；
    同时过滤器不得误杀相关文档（filter-aware recall 达标）。
    """

    def test_filter_cases_exist(self):
        assert load_filter_cases(), "golden/filter_cases.jsonl 为空，过滤评估层未接线"

    def test_zero_leakage_per_case(self, report):
        for cid, m in report.filter_eval["per_case"].items():
            assert m["leakage_rate"] == 0.0, (
                f"filter case {cid} 泄漏率 {m['leakage_rate']:.4f} > 0，"
                "search() 返回了违反过滤条件的结果"
            )

    def test_zero_trap_hits_per_case(self, report):
        for cid, m in report.filter_eval["per_case"].items():
            assert m["trap_hits"] == 0.0, (
                f"filter case {cid} 命中 trap 文档 {m['trap_hits']:.0f} 个，"
                "主题相关但应被过滤排除的文档泄漏到结果中"
            )

    def test_filter_aware_recall_floor(self, report):
        # 过滤器不得误杀：有正例标注的 case，相关文档至少召回一半
        for cid, m in report.filter_eval["per_case"].items():
            case = next(c for c in load_filter_cases() if c["case_id"] == cid)
            if any(g >= 1 for g in case["relevant_docs"].values()):
                assert m["recall"] >= 0.5, (
                    f"filter case {cid} recall={m['recall']:.4f} < 0.5，"
                    "疑似过滤条件误杀相关文档"
                )


@pytest.mark.eval
class TestGateEval:
    """Step 5 质量门控评估回归锁（golden/gate_cases.jsonl）。

    强制 case（无 known_limitation 标记）：库内查询不误拦、分数门控
    （decide_quality_gate 喂合成分数，含阈值边界）决策正确、拦截诊断字段完整。
    已知局限 case（g03-g05 零重叠库外查询拦不住）：不纳入指标，但棘轮断言 ——
    一旦修复翻绿，必须摘标纳入强制锁（防“锁空转”：局限标签不是永久豁免）。
    """

    def test_gate_cases_exist(self):
        assert load_gate_cases(), "golden/gate_cases.jsonl 为空，门控评估层未接线"

    def test_gate_accuracy_full(self, report):
        per_case = report.gate_eval["per_case"]
        enforced = [c["case_id"] for c in load_gate_cases()
                    if not c.get("known_limitation")]
        wrong = [cid for cid in enforced if not per_case[cid]["correct"]]
        assert not wrong, (
            f"门控决策错误 case: {wrong}（明细 {[(c, per_case[c]) for c in wrong]}）"
        )
        assert report.gate_eval["macro"]["accuracy"] == 1.0

    def test_known_limitation_ratchet(self, report):
        # 棘轮：已知局限 case 一旦修复翻绿，必须摘掉 known_limitation 标记
        # 纳入强制锁并重跑 --update-baseline（标签不是永久豁免）
        per_case = report.gate_eval["per_case"]
        fixed = [cid for cid in report.gate_eval.get("known_limitation_cases", [])
                 if per_case[cid]["correct"]]
        assert not fixed, (
            f"已知局限 case {fixed} 已修复翻绿：请摘掉 gate_cases.jsonl 中的 "
            "known_limitation 标记纳入强制锁，并重跑 --update-baseline"
        )

    def test_no_false_block_no_false_pass(self, report):
        gm = report.gate_eval["macro"]
        assert gm["false_block_rate"] == 0.0, "库内/高分 case 被误拦（false block）"
        assert gm["false_pass_rate"] == 0.0, "库外/低分 case 被误放（false pass）"

    def test_blocked_diagnostics_complete(self, report):
        assert report.gate_eval["macro"]["diagnostics_complete"] == 1.0, (
            "存在拦截诊断字段不完整的 case（需 stage/reason/query/threshold，"
            "quality_gate 阶段额外需 top_score/top_text_preview）"
        )

    def test_gate_baseline_lock(self, report, baseline):
        base_gm = baseline.get("gate", {}).get("macro")
        assert base_gm, "baseline.json 缺 gate 段，请重跑 --update-baseline"
        cur_gm = report.gate_eval["macro"]
        for m in ("accuracy", "diagnostics_complete"):
            assert cur_gm[m] >= base_gm[m] - DEFAULT_TOL, (
                f"gate.{m} 回归：当前 {cur_gm[m]:.4f} < baseline {base_gm[m]:.4f}"
            )
