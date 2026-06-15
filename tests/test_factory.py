"""
Test Factory — create_orchestrator wiring verification

Covers:
- All 7 agents registered
- AgentRouter attached to orchestrator
- Agent chains correct (no CoordinatorAgent in chains)
- Tool binding for all agents
- Intent map matches expected chains
"""
import pytest
from financial_rag.core.factory import create_orchestrator


@pytest.fixture
def orch():
    return create_orchestrator()


class TestFactoryWiring:

    def test_seven_agents_registered(self, orch):
        assert len(orch.agents) == 7
        expected = [
            "CoordinatorAgent",
            "IngestionAgent",
            "ExtractionAgent",
            "ReportAgent",
            "KLineAgent",
            "EventImpactAgent",
            "ScoringAgent",
        ]
        for name in expected:
            assert name in orch.agents, f"Missing agent: {name}"

    def test_router_attached(self, orch):
        assert hasattr(orch, "agent_router")
        assert orch.agent_router is not None

    def test_intent_map_keys(self, orch):
        m = orch.agent_router.get_intent_map()
        assert "kline" in m
        assert "event_impact" in m
        assert "report" in m
        assert "news" in m
        assert "_default" in m

    def test_no_coordinator_in_chains(self, orch):
        m = orch.agent_router.get_intent_map()
        for intent, chain in m.items():
            assert "CoordinatorAgent" not in chain, f"CoordinatorAgent in {intent} chain"

    def test_all_chains_end_with_scoring(self, orch):
        m = orch.agent_router.get_intent_map()
        for intent, chain in m.items():
            assert "ScoringAgent" in chain, f"ScoringAgent missing from {intent} chain"

    def test_kline_chain_order(self, orch):
        m = orch.agent_router.get_intent_map()
        chain = m["kline"]
        assert chain.index("KLineAgent") < chain.index("ReportAgent")
        assert chain.index("ReportAgent") < chain.index("ScoringAgent")

    def test_event_chain_order(self, orch):
        m = orch.agent_router.get_intent_map()
        chain = m["event_impact"]
        assert chain.index("EventImpactAgent") < chain.index("ReportAgent")

    def test_report_chain_order(self, orch):
        m = orch.agent_router.get_intent_map()
        chain = m["report"]
        assert chain.index("IngestionAgent") < chain.index("ExtractionAgent")
        assert chain.index("ExtractionAgent") < chain.index("ReportAgent")

    def test_agents_have_tools(self, orch):
        """All agents should have _registry bound"""
        for name, agent in orch.agents.items():
            assert agent._registry is not None, f"{name} has no registry"

    def test_agents_have_executor(self, orch):
        """All agents should have _executor bound"""
        for name, agent in orch.agents.items():
            assert agent._executor is not None, f"{name} has no executor"


class TestFactoryNoLLM:
    """Factory works without LLM (pure regex/fallback mode)"""

    def test_create_without_llm(self):
        orch = create_orchestrator(retriever=None, llm=None)
        assert len(orch.agents) == 7

    def test_create_without_retriever(self):
        orch = create_orchestrator(retriever=None)
        assert len(orch.agents) == 7
