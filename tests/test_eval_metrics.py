"""
L1 指标纯函数单测 — 手算精确值验证（TDD oracle）。

每条断言的期望值均由手算得出（见注释），独立于实现，确保
recall / precision / mrr / ndcg / map 的数学定义正确无误。
对应生产代码：financial_rag/eval/metrics.py
"""
import pytest

from financial_rag.eval.metrics import (
    recall_at_k, precision_at_k, mrr, ndcg_at_k, map_at_k,
    evaluate_query, dedupe_ranked, relevant_ids, METRIC_NAMES,
)


@pytest.mark.unit
class TestRelevantIds:
    def test_min_grade_1(self):
        assert relevant_ids({"A": 3, "B": 1, "C": 0}) == {"A", "B"}

    def test_min_grade_2(self):
        assert relevant_ids({"A": 3, "B": 1, "C": 0}, min_grade=2) == {"A"}


@pytest.mark.unit
class TestRecallPrecision:
    def test_recall_partial(self):
        # top-3=[A,B,C] 命中 {A,C}=2，rel 总数=3 → 2/3
        assert recall_at_k(["A", "B", "C", "D"], {"A", "C", "E"}, 3) == pytest.approx(2 / 3)

    def test_recall_full(self):
        assert recall_at_k(["A", "B"], {"A", "B"}, 5) == 1.0

    def test_recall_empty_rel(self):
        assert recall_at_k(["A"], set(), 3) == 0.0

    def test_precision_half(self):
        # top-4 命中 {A,C}=2 / 4 = 0.5
        assert precision_at_k(["A", "B", "C", "D"], {"A", "C"}, 4) == 0.5

    def test_precision_k0(self):
        assert precision_at_k(["A"], {"A"}, 0) == 0.0


@pytest.mark.unit
class TestMRR:
    def test_first_at_rank3(self):
        assert mrr(["X", "Y", "A"], {"A"}) == pytest.approx(1 / 3)

    def test_first_at_rank1(self):
        assert mrr(["A", "X"], {"A"}) == 1.0

    def test_no_hit(self):
        assert mrr(["X", "Y", "Z"], {"A"}) == 0.0


@pytest.mark.unit
class TestNDCG:
    def test_perfect_single(self):
        # DCG=(2^3-1)/log2(2)=7/1=7，IDCG=7 → 1.0
        assert ndcg_at_k(["A"], {"A": 3}, 1) == pytest.approx(1.0)

    def test_graded_wrong_order(self):
        # 手算: DCG  = 1/log2(2) + 7/log2(3) = 1.0   + 4.4165 = 5.4165
        #       IDCG = 7/log2(2) + 1/log2(3) = 7.0   + 0.6309 = 7.6309
        #       NDCG = 5.4165 / 7.6309 = 0.7098
        val = ndcg_at_k(["A", "B"], {"A": 1, "B": 3}, 2)
        assert val == pytest.approx(0.7098, abs=1e-3)
        # 高相关排后面必须被惩罚：错序 < 正序
        assert val < ndcg_at_k(["B", "A"], {"A": 1, "B": 3}, 2)

    def test_graded_right_order_is_1(self):
        # ranked=[B(g3),A(g1)] 完美降序 → NDCG=1.0
        assert ndcg_at_k(["B", "A"], {"A": 1, "B": 3}, 2) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k(["A", "B"], {"A": 0}, 2) == 0.0


@pytest.mark.unit
class TestMAP:
    def test_ap_interleaved(self):
        # ranked=[A,X,B] rel={A,B}: prec@1=1/1, prec@3=2/3 → (1+0.667)/2=0.8333
        assert map_at_k(["A", "X", "B"], {"A", "B"}, 3) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_ap_perfect(self):
        assert map_at_k(["A", "B"], {"A", "B"}, 2) == 1.0

    def test_ap_empty_rel(self):
        assert map_at_k(["A"], set(), 1) == 0.0


@pytest.mark.unit
class TestDedupe:
    def test_dedupe_keeps_first_rank(self):
        items = [{"meta": {"doc_id": "A"}}, {"meta": {"doc_id": "A"}}, {"meta": {"doc_id": "B"}}]
        assert dedupe_ranked(items) == ["A", "B"]

    def test_dedupe_skips_none(self):
        items = [{"meta": {}}, {"meta": {"doc_id": "A"}}]
        assert dedupe_ranked(items) == ["A"]


@pytest.mark.unit
class TestEvaluateQuery:
    def test_returns_all_metrics(self):
        scores = evaluate_query(["A", "B"], {"A": 3, "B": 1}, 5)
        assert set(scores.keys()) == set(METRIC_NAMES)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_empty_ranked_all_zero(self):
        scores = evaluate_query([], {"A": 3}, 5)
        assert all(v == 0.0 for v in scores.values())


@pytest.mark.unit
class TestGradeZeroNegatives:
    """grade=0 显式负例（错误 case）必须被当作"不相关"。

    强化语料库后 qrels 引入 grade=0 负例，指标层须保证：
    负例不计入召回/命中，且负例占据 top-k 位置会拉低 precision/NDCG/MAP。
    """

    def test_relevant_ids_excludes_grade_zero(self):
        assert relevant_ids({"A": 3, "B": 1, "C": 0}) == {"A", "B"}

    def test_recall_ignores_grade_zero(self):
        rel = relevant_ids({"A": 3, "B": 0})
        # 只命中负例 B → recall=0（B 不是正例）
        assert recall_at_k(["B"], rel, 5) == 0.0
        # 命中正例 A → recall=1/1
        assert recall_at_k(["B", "A"], rel, 5) == 1.0

    def test_precision_penalized_by_negative_in_topk(self):
        rel = relevant_ids({"A": 3, "B": 0})
        # top-2=[B(负), A(正)]：负例占位 → precision=1/2
        assert precision_at_k(["B", "A"], rel, 2) == 0.5

    def test_ndcg_excludes_grade_zero_from_idcg(self):
        grades = {"A": 3, "B": 0}
        # ranked=[A]：IDCG 只按 A(g3) 算 → NDCG=1.0（B 的 g0 不进 IDCG）
        assert ndcg_at_k(["A"], grades, 2) == pytest.approx(1.0)
        # ranked=[B,A]：B(g0) gain=0 不贡献但占 rank1，把 A 挤到 rank2
        # DCG=7/log2(3)=4.4165, IDCG=7/log2(2)=7 → 0.6309（负例占位被惩罚）
        assert ndcg_at_k(["B", "A"], grades, 2) == pytest.approx(0.6309, abs=1e-3)

    def test_map_ignores_grade_zero(self):
        rel = relevant_ids({"A": 3, "B": 0})
        # ranked=[B,A]：A 在 rank2 命中 → prec@2=1/2, AP=(1/2)/min(1,2)=0.5
        assert map_at_k(["B", "A"], rel, 2) == pytest.approx(0.5)
        # ranked=[A,B]：A 在 rank1 命中 → AP=1.0
        assert map_at_k(["A", "B"], rel, 2) == pytest.approx(1.0)
