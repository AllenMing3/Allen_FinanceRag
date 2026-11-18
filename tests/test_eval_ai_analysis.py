"""
Test 质量评估 AI 预分析（eval/ai_analysis.py）— 辅助观测层的离线锁。

定位：digest 纯函数离线确定 + analyze 失败语义（辅助层不阻断）。
全部用合成 report / stub caller，不跑真实 LLM、不触网。
"""
import json

import pytest

from financial_rag.eval import ai_analysis
from financial_rag.eval.ai_analysis import (
    analyze,
    build_digest_l1,
    build_digest_l2,
    format_analysis,
    save_analysis,
)
from financial_rag.eval.output_runner import Comparison as L2Comparison
from financial_rag.eval.output_runner import OutputReport
from financial_rag.eval.runner import ChannelResult, Comparison, EvalReport


# ===================== 合成数据 =====================

def _l1_report():
    fusion = ChannelResult(
        macro={"recall": 0.8, "precision": 0.6, "mrr": 0.7, "ndcg": 0.75, "map": 0.65},
        per_query={
            "q_good": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0},
            "q_bad": {"recall": 0.2, "mrr": 0.1, "ndcg": 0.15},
            "q_mid": {"recall": 0.6, "mrr": 0.5, "ndcg": 0.55},
        },
        per_tag={"bug_scene": {"recall": 0.5, "mrr": 0.4, "ndcg": 0.45}},
    )
    return EvalReport(
        top_k=5, query_count=3, doc_count=4, chunk_count=6,
        channels={"fusion": fusion},
        filter_eval={"case_count": 2, "macro": {"leakage_rate": 0.0, "recall": 1.0}},
        gate_eval={"case_count": 2, "threshold": 0.3,
                   "macro": {"accuracy": 0.5, "false_block_rate": 0.0,
                             "false_pass_rate": 0.5, "diagnostics_complete": 1.0},
                   "per_case": {"g_ok": {"correct": True, "blocked": True},
                                "g_miss": {"correct": False, "blocked": False,
                                           "stage": None}},
                   "known_limitation_cases": ["g_known"]},
    )


def _l1_cmp():
    return Comparison(
        deltas={("fusion", "recall"): -0.05, ("gate", "accuracy"): -0.1},
        regressions=[{"channel": "fusion", "metric": "recall",
                      "baseline": 0.85, "current": 0.8, "delta": -0.05}],
        tol=0.02,
    )


def _l2_report():
    return OutputReport(
        case_count=3, good_count=1, bad_count=2,
        macro={"numerical_fidelity": 0.8, "source_grounding": 0.7,
               "field_completeness": 0.9, "output_quality": 0.78},
        per_tier={"good": {"output_quality": 0.95}, "bad": {"output_quality": 0.6}},
        per_tag={"fabricated_number": {"output_quality": 0.5}},
        separation_gap=0.35,
        per_case={
            "c_good": {"output_quality": 0.95, "numerical_fidelity": 1.0,
                       "source_grounding": 0.9, "field_completeness": 1.0,
                       "unmatched_numbers": [], "unanchored": [], "missing_fields": []},
            "c_fab": {"output_quality": 0.4, "numerical_fidelity": 0.2,
                      "source_grounding": 0.6, "field_completeness": 1.0,
                      "unmatched_numbers": ["999.99亿元"] * 8,
                      "unanchored": ["无据句子" * 40], "missing_fields": []},
            "c_miss": {"output_quality": 0.6, "numerical_fidelity": 1.0,
                       "source_grounding": 0.8, "field_completeness": 0.3,
                       "unmatched_numbers": [], "unanchored": [],
                       "missing_fields": ["outlook"]},
        },
    )


class _StubCaller:
    """记录调用参数的 call_json stub（不触网、不进 LLMCaller 进程级缓存）。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def call_json(self, messages, system=None, **kwargs):
        self.calls.append({"messages": messages, "system": system, **kwargs})
        if self.error:
            raise self.error
        return self.result


_VALID_RESULT = {
    "overall": "regressed",
    "summary": "fusion recall 回退超阈值",
    "findings": [{"severity": "high", "finding": "recall 回退",
                  "evidence": "0.85 -> 0.8"}],
    "possible_causes": ["切分策略变更"],
    "next_steps": ["对比最近 chunker 改动"],
}


# ===================== digest 纯函数 =====================

@pytest.mark.unit
class TestBuildDigestL1:

    def test_regressions_and_deltas_included(self):
        d = build_digest_l1(_l1_report(), _l1_cmp())
        bc = d["baseline_comparison"]
        assert bc["regressions"][0]["metric"] == "recall"
        assert bc["deltas"]["fusion.recall"] == -0.05   # 元组键序列化
        assert bc["deltas"]["gate.accuracy"] == -0.1

    def test_low_recall_queries_sorted_and_limited(self):
        d = build_digest_l1(_l1_report())
        lows = d["low_recall_queries"]
        assert lows[0]["query_id"] == "q_bad"           # recall 最低排最前
        assert [q["query_id"] for q in lows] == ["q_bad", "q_mid", "q_good"]

    def test_gate_failed_cases_surface(self):
        d = build_digest_l1(_l1_report())
        assert d["gate"]["failed_cases"] == [
            {"case_id": "g_miss", "blocked": False, "stage": None}]
        assert d["gate"]["known_limitation_cases"] == ["g_known"]

    def test_no_baseline_marked_none(self):
        d = build_digest_l1(_l1_report(), None)
        assert d["baseline_comparison"] is None

    def test_offline_deterministic(self):
        r, c = _l1_report(), _l1_cmp()
        assert build_digest_l1(r, c) == build_digest_l1(r, c)


@pytest.mark.unit
class TestBuildDigestL2:

    def test_low_cases_sorted_with_meta_and_evidence(self):
        meta = {"c_fab": {"quality_tier": "bad", "tags": ["fabricated_number"]}}
        d = build_digest_l2(_l2_report(), case_meta=meta)
        lows = d["low_quality_cases"]
        assert lows[0]["case_id"] == "c_fab"            # output_quality 最低排最前
        assert lows[0]["tags"] == ["fabricated_number"]
        assert lows[0]["quality_tier"] == "bad"
        assert lows[0]["unmatched_numbers"]             # 编造数字证据入选

    def test_evidence_truncated(self):
        d = build_digest_l2(_l2_report())
        fab = d["low_quality_cases"][0]
        assert len(fab["unmatched_numbers"]) <= 5       # 列表截断
        assert len(fab["unanchored_preview"][0]) <= ai_analysis._TEXT_PREVIEW_LIMIT + 1
        assert fab["unanchored_count"] == 1             # 计数保留全量信息

    def test_l2_comparison_gap_delta_included(self):
        cmp = L2Comparison(deltas={"output_quality": -0.03}, gap_delta=-0.05,
                           regressions=[{"scope": "macro", "metric": "output_quality",
                                         "baseline": 0.8, "current": 0.77, "delta": -0.03}],
                           tol=0.02)
        d = build_digest_l2(_l2_report(), cmp)
        bc = d["baseline_comparison"]
        assert bc["gap_delta"] == -0.05
        assert bc["deltas"]["output_quality"] == -0.03
        assert bc["regressions"][0]["scope"] == "macro"

    def test_separation_gap_and_tiers(self):
        d = build_digest_l2(_l2_report())
        assert d["separation_gap"] == 0.35
        assert d["per_tier"]["good"]["output_quality"] == 0.95


# ===================== analyze 失败语义（辅助层不阻断） =====================

@pytest.mark.unit
class TestAnalyze:

    def test_valid_result_passthrough_no_cache(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "_has_api_key", lambda: True)
        stub = _StubCaller(result=dict(_VALID_RESULT))
        out = analyze(build_digest_l1(_l1_report(), _l1_cmp()), caller=stub)
        assert out == _VALID_RESULT
        assert stub.calls[0]["use_cache"] is False      # 防进程级缓存串扰
        assert stub.calls[0]["temperature"] == 0.0
        assert "l1_retrieval" in stub.calls[0]["messages"]

    def test_parse_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "_has_api_key", lambda: True)
        stub = _StubCaller(result=None)                 # call_json 解析耗尽返回 None
        assert analyze({"kind": "l2_output"}, caller=stub) is None

    def test_schema_violation_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "_has_api_key", lambda: True)
        stub = _StubCaller(result={"unexpected": "shape"})
        assert analyze({"kind": "l2_output"}, caller=stub) is None

    def test_call_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "_has_api_key", lambda: True)
        stub = _StubCaller(error=RuntimeError("API 挂了"))
        assert analyze({"kind": "l1_retrieval"}, caller=stub) is None

    def test_missing_api_key_skips_without_call(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "_has_api_key", lambda: False)
        stub = _StubCaller(result=dict(_VALID_RESULT))
        assert analyze({"kind": "l1_retrieval"}, caller=stub) is None
        assert stub.calls == []                         # 缺 key 不发起调用


# ===================== 格式化与落盘 =====================

@pytest.mark.unit
def test_format_analysis_sections():
    text = format_analysis(_VALID_RESULT)
    assert "存在回归" in text
    assert "fusion recall 回退超阈值" in text
    assert "0.85 -> 0.8" in text
    assert "对比最近 chunker 改动" in text


@pytest.mark.integration
def test_save_analysis_payload(tmp_path):
    path = tmp_path / "ai_analysis_results.json"
    save_analysis(_VALID_RESULT, "l1_retrieval", path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "l1_retrieval"
    assert payload["role"] == "auxiliary"
    assert payload["overall"] == "regressed"
    assert "timestamp" in payload