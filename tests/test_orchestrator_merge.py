"""
Test Orchestrator _apply_updates merge behavior

Covers:
- dict attributes (metadata) are merged, not replaced
- list attributes (intermediate_findings) are extended, not replaced
- scalar attributes (final_answer) are replaced
- Unknown keys go to metadata
- Full multi-agent chain simulation
"""
import pytest
from financial_rag.core.base import AgentContext, AgentResult
from financial_rag.core.orchestrator import AgentOrchestrator, CoordinatorConfig


@pytest.fixture
def orch():
    """Orchestrator with verbose off"""
    return AgentOrchestrator(CoordinatorConfig(verbose=False))


@pytest.fixture
def ctx():
    """AgentContext with initial pipeline data"""
    return AgentContext(
        raw_input="test query",
        metadata={
            "retrieved_items": [{"text": "doc1"}],
            "fetched_data": [{"x": 1}],
            "intent": "kline",
        },
    )


class TestDictMerge:
    """dict-type attributes merge (not replace)"""

    def test_metadata_merge_preserves_existing(self, orch, ctx):
        orch.context = ctx
        r = AgentResult(
            success=True, agent_name="A1",
            context_updates={"metadata": {"ts_code": "600519.SH"}},
        )
        orch._apply_updates(r)

        m = orch.context.metadata
        assert m["retrieved_items"] == [{"text": "doc1"}]  # preserved
        assert m["fetched_data"] == [{"x": 1}]  # preserved
        assert m["intent"] == "kline"  # preserved
        assert m["ts_code"] == "600519.SH"  # new key added

    def test_metadata_merge_overwrites_key(self, orch, ctx):
        """Same key gets overwritten within metadata"""
        orch.context = ctx
        r = AgentResult(
            success=True, agent_name="A1",
            context_updates={"metadata": {"intent": "event_impact"}},
        )
        orch._apply_updates(r)
        assert orch.context.metadata["intent"] == "event_impact"

    def test_multi_agent_metadata_accumulates(self, orch, ctx):
        """Multiple agents' metadata all visible at end"""
        orch.context = ctx

        # Agent 1: routing
        r1 = AgentResult(
            success=True, agent_name="Router",
            context_updates={"metadata": {"ts_code": "600519.SH", "confidence": 0.9}},
        )
        orch._apply_updates(r1)

        # Agent 2: analysis
        r2 = AgentResult(
            success=True, agent_name="Analyzer",
            context_updates={"metadata": {"analysis_score": 0.85}},
        )
        orch._apply_updates(r2)

        # Agent 3: report
        r3 = AgentResult(
            success=True, agent_name="Reporter",
            context_updates={"metadata": {"report_count": 3}},
        )
        orch._apply_updates(r3)

        m = orch.context.metadata
        # All agents' metadata preserved
        assert m["retrieved_items"] == [{"text": "doc1"}]
        assert m["ts_code"] == "600519.SH"
        assert m["confidence"] == 0.9
        assert m["analysis_score"] == 0.85
        assert m["report_count"] == 3


class TestListExtend:
    """list-type attributes extend (not replace)"""

    def test_intermediate_findings_extend(self, orch, ctx):
        ctx.intermediate_findings = [{"stage": "fetch", "ok": True}]
        orch.context = ctx

        r = AgentResult(
            success=True, agent_name="A1",
            context_updates={"intermediate_findings": [{"stage": "kline_analysis"}]},
        )
        orch._apply_updates(r)

        findings = orch.context.intermediate_findings
        assert len(findings) == 2
        assert findings[0]["stage"] == "fetch"
        assert findings[1]["stage"] == "kline_analysis"

    def test_multi_agent_findings_chain(self, orch, ctx):
        """Full chain: each agent appends its findings"""
        ctx.intermediate_findings = []
        orch.context = ctx

        for stage in ["coordination", "kline_analysis", "report", "scoring"]:
            r = AgentResult(
                success=True, agent_name=stage,
                context_updates={"intermediate_findings": [{"stage": stage}]},
            )
            orch._apply_updates(r)

        assert len(orch.context.intermediate_findings) == 4
        stages = [f["stage"] for f in orch.context.intermediate_findings]
        assert stages == ["coordination", "kline_analysis", "report", "scoring"]


class TestScalarReplace:
    """scalar attributes replace (not merge)"""

    def test_final_answer_replace(self, orch, ctx):
        ctx.final_answer = "old answer"
        orch.context = ctx

        r = AgentResult(
            success=True, agent_name="A1",
            context_updates={"final_answer": "new answer"},
        )
        orch._apply_updates(r)
        assert orch.context.final_answer == "new answer"

    def test_final_answer_from_none(self, orch, ctx):
        ctx.final_answer = None
        orch.context = ctx

        r = AgentResult(
            success=True, agent_name="A1",
            context_updates={"final_answer": "first answer"},
        )
        orch._apply_updates(r)
        assert orch.context.final_answer == "first answer"


class TestUnknownKeys:
    """Unknown context_updates keys go to metadata"""

    def test_unknown_key_to_metadata(self, orch, ctx):
        orch.context = ctx
        r = AgentResult(
            success=True, agent_name="A1",
            context_updates={"custom_field": "custom_value"},
        )
        orch._apply_updates(r)
        assert orch.context.metadata["custom_field"] == "custom_value"


class TestEmptyUpdates:
    """Edge cases: empty updates, None"""

    def test_empty_context_updates(self, orch, ctx):
        orch.context = ctx
        r = AgentResult(success=True, agent_name="A1", context_updates={})
        orch._apply_updates(r)
        assert orch.context.metadata["intent"] == "kline"

    def test_none_context_updates(self, orch, ctx):
        orch.context = ctx
        r = AgentResult(success=True, agent_name="A1")
        orch._apply_updates(r)
        assert orch.context.metadata["intent"] == "kline"
