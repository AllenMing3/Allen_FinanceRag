"""
L1 检索层评估回归锁 — 把 runner.evaluate() 的数字变成 CI 护栏。

对应 plan E（接线）+ Test Plan #3/#5，全部 @pytest.mark.eval：
  1. 离线确定性：evaluate() 连跑两次 to_dict() 完全一致（无网络、无随机）
  2. fusion 回归锁：fusion 通道 recall/mrr/ndcg >= baseline - tol，回退即失败
  3. bug_scene 达标：今天修的 bug 场景子集（未登录词 / 多 chunk 长文）不低于 baseline，
     并有绝对底线（recall >= 0.5）证明修复确实生效，而非仅"没变差"
  4. 分通道可见性：bm25/vector/fusion 三通道独立非零、指标合法，不静默坍塌
     （Step3 "Jaccard 融合淹没 BM25 改善" 教训的制度化护栏）

快跑通道用 `-m "not eval"` 排除；显式跑：pytest tests/test_eval_retrieval.py -m eval
baseline.json 由 `python -m financial_rag.eval.runner --update-baseline` 生成并纳入 git。
"""
import pytest

from financial_rag.eval.metrics import METRIC_NAMES
from financial_rag.eval.runner import (
    CHANNELS, DEFAULT_TOP_K, DEFAULT_TOL, GUARDED_METRICS,
    evaluate, load_baseline,
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
