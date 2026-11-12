"""
ReliabilityService 统一入口 + 静默跳过堵口 — P1-2 回归锁。

钉死三件事（路线图方向二）：
1. 入口收敛：所有调用点经 ReliabilityService，source 归一化 + mode 显式传参，
   同一输入必得同一分数（确定性）。
2. 静默跳过 → 显式 skipped：空 answer / 空 sources 不再返回 {} 或 None，
   而是带 skip_reason 的 skipped 结构，序列化后前端可见「未校验」。
3. Guard 引擎不动：本文件不测 L1-L4 规则本身（已有覆盖），只测接线契约。
"""
import pytest

from financial_rag.guard.service import (
    ReliabilityService,
    normalize_sources,
    SKIP_EMPTY_ANSWER,
    SKIP_NO_SOURCES,
)
from financial_rag.guard.serializer import serialize_guard_result

SOURCE_TEXT = ("商汤科技2024年实现营业收入50.3亿元，同比增长36.4%。"
               "生成式AI业务收入占比达60%，成为公司核心增长引擎。")
ANSWER = "商汤科技2024年营收50.3亿元，同比增长36.4%[1]。"


@pytest.mark.unit
class TestNormalizeSources:
    def test_raw_items_strip_meta(self):
        raw = [{"text": SOURCE_TEXT, "meta": {"source": "kb"}, "score": 0.9}]
        assert normalize_sources(raw) == [{"text": SOURCE_TEXT}]

    def test_mixed_str_and_dict(self):
        assert normalize_sources([SOURCE_TEXT, {"text": "第二条证据。"}]) == [
            {"text": SOURCE_TEXT}, {"text": "第二条证据。"}]

    def test_empty_text_filtered(self):
        assert normalize_sources([{"text": ""}, {"text": "   "}, ""]) == []

    def test_none_returns_empty(self):
        assert normalize_sources(None) == []


@pytest.mark.unit
class TestServiceSkipped:
    def test_empty_answer_explicit_skipped(self):
        out = ReliabilityService().check("", [{"text": SOURCE_TEXT}])
        assert out["skipped"] is True
        assert out["skip_reason"] == SKIP_EMPTY_ANSWER
        assert out["passed"] is False
        assert out["checks"] == {}

    def test_whitespace_answer_skipped(self):
        out = ReliabilityService().check("   \n  ", [{"text": SOURCE_TEXT}])
        assert out["skipped"] is True
        assert out["skip_reason"] == SKIP_EMPTY_ANSWER

    def test_no_sources_explicit_skipped(self):
        out = ReliabilityService().check(ANSWER, [])
        assert out["skipped"] is True
        assert out["skip_reason"] == SKIP_NO_SOURCES

    def test_all_empty_sources_skipped(self):
        out = ReliabilityService().check(ANSWER, [{"text": ""}, {"meta": {}}])
        assert out["skipped"] is True
        assert out["skip_reason"] == SKIP_NO_SOURCES


@pytest.mark.unit
class TestServiceCheck:
    def test_normal_returns_raw_checks(self):
        out = ReliabilityService().check(ANSWER, [{"text": SOURCE_TEXT}])
        assert out["skipped"] is False
        assert out["skip_reason"] == ""
        for layer in ("L1_source_grounding", "L2_numerical_fidelity",
                      "L3_citation_integrity", "L4_structure_compliance"):
            assert layer in out["checks"]
        assert 0.0 <= out["overall_score"] <= 1.0

    def test_deterministic_same_input_same_score(self):
        svc = ReliabilityService()
        a = svc.check(ANSWER, [{"text": SOURCE_TEXT}])
        b = svc.check(ANSWER, [{"text": SOURCE_TEXT}])
        assert a["overall_score"] == b["overall_score"]
        assert a["checks"] == b["checks"]

    def test_mode_passthrough(self):
        """analysis 与 rag 模式对同一文本结构分不同 —— mode 必须真实透传。"""
        text = "【关键信号】商汤营收增长。【风险提示】行业竞争加剧。"
        svc = ReliabilityService()
        ana = svc.check(text, [{"text": SOURCE_TEXT}], mode="analysis")
        rag = svc.check(text, [{"text": SOURCE_TEXT}], mode="rag")
        assert (ana["checks"]["L4_structure_compliance"]["score"]
                > rag["checks"]["L4_structure_compliance"]["score"])

    def test_serialized_integration(self):
        out = ReliabilityService().check(ANSWER, [{"text": SOURCE_TEXT}])
        ser = serialize_guard_result(out)
        assert ser["skipped"] is False
        assert ser["layers"]["L1_source_grounding"]["score"] >= 0


@pytest.mark.unit
class TestSerializerSkipped:
    def test_skipped_shape_preserved(self):
        out = ReliabilityService().check("", [{"text": SOURCE_TEXT}])
        ser = serialize_guard_result(out)
        assert ser["skipped"] is True
        assert ser["skip_reason"] == SKIP_EMPTY_ANSWER
        assert ser["layers"] == {}
        assert ser["passed"] is False


@pytest.mark.unit
class TestCallSites:
    def test_check_hallucination_empty_text_skipped_not_empty_dict(self):
        """scoring_tools 空文本曾 return {} 无痕消失 → 现在显式 skipped。"""
        from financial_rag.tools.scoring_tools import check_hallucination
        out = check_hallucination("")
        assert out["skipped"] is True
        assert out["skip_reason"] == SKIP_EMPTY_ANSWER

    def test_check_hallucination_normal(self):
        from financial_rag.tools.scoring_tools import check_hallucination
        out = check_hallucination(ANSWER, [{"text": SOURCE_TEXT}])
        assert out["skipped"] is False
        assert out["checks"]["L1_source_grounding"]["score"] >= 0

    def test_scoring_agent_can_handle_empty_context(self):
        """can_handle 兜底：空上下文也运行评分，产出显式未校验而非整层消失。"""
        from financial_rag.agents.scoring_agent import ScoringAgent
        from financial_rag.core.base import AgentContext
        assert ScoringAgent().can_handle(AgentContext()) is True

    def test_analysis_router_empty_result_injects_skipped(self):
        from financial_rag.api.analysis_router import _run_hallucination_check
        result = {}
        _run_hallucination_check(result)
        assert result["hallucination"]["skipped"] is True
        assert result["hallucination"]["skip_reason"] == SKIP_EMPTY_ANSWER

    def test_analysis_router_no_sources_injects_skipped(self):
        from financial_rag.api.analysis_router import _run_hallucination_check
        result = {"analysis": "商汤营收增长。"}
        _run_hallucination_check(result)
        assert result["hallucination"]["skipped"] is True
        assert result["hallucination"]["skip_reason"] == SKIP_NO_SOURCES

    def test_analysis_router_normal_injects_layers(self):
        from financial_rag.api.analysis_router import _run_hallucination_check
        result = {"analysis": "商汤科技2024年营收50.3亿元。",
                  "kb_sources": [{"text": SOURCE_TEXT}]}
        _run_hallucination_check(result)
        hal = result["hallucination"]
        assert hal["skipped"] is False
        assert "L1_source_grounding" in hal["layers"]

    def test_score_report_skipped_branch(self):
        """skipped 时报告显式说「未校验」，不再误印 0% 可信度。"""
        from financial_rag.tools.scoring_tools import generate_score_report
        out = generate_score_report(
            pipeline_scores={"grade": "B", "stages": [], "summary": {}},
            hallucination_check={"skipped": True, "skip_reason": SKIP_NO_SOURCES},
            query="测试",
        )
        assert "未校验" in out["report"]
        assert "可信度" not in out["report"]