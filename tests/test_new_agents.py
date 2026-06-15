"""
Test new/refactored agents — can_handle + process with tool delegation

Covers:
- CoordinatorAgent: can_handle (always True), process (intent classification + chain selection)
- KLineAgent: can_handle (intent/stock/keyword), process (tool delegation)
- EventImpactAgent: can_handle (intent/date/keyword), process (missing params)
- ScoringAgent: can_handle (findings/final_answer), process (scoring pipeline)
- ReportAgent: can_handle (specialist findings), _findings_to_documents
"""
import pytest
from financial_rag.core.base import AgentContext, AgentResult
from financial_rag.tools.core import FunctionRegistry, ToolExecutor, create_financial_registry


@pytest.fixture
def tools():
    registry = create_financial_registry(retriever=None, llm=None)
    executor = ToolExecutor(registry)
    return registry, executor


def _bind(agent, tools):
    registry, executor = tools
    agent.bind_tools(registry, executor)
    return agent


# ===================== CoordinatorAgent =====================


class TestCoordinatorAgent:

    def test_can_handle_always_true(self):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = CoordinatorAgent()
        ctx = AgentContext(raw_input="anything")
        assert agent.can_handle(ctx) is True

    def test_process_kline_query(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = _bind(CoordinatorAgent(), tools)
        ctx = AgentContext(raw_input="茅台走势分析")
        result = agent.run(ctx)

        assert result.success
        assert result.agent_name == "CoordinatorAgent"
        data = result.data
        assert data["intent"] == "kline"
        assert "KLineAgent" in data["agent_chain"]

    def test_process_event_query(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = _bind(CoordinatorAgent(), tools)
        ctx = AgentContext(raw_input="2024年6月1日发生了什么大事")
        result = agent.run(ctx)

        assert result.success
        data = result.data
        assert data["intent"] == "event_impact"

    def test_metadata_injection(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = _bind(CoordinatorAgent(), tools)
        ctx = AgentContext(raw_input="2024-06-01 茅台发生了什么")
        result = agent.run(ctx)

        assert result.success
        meta = result.context_updates.get("metadata", {})
        assert meta.get("intent") is not None

    def test_extract_metadata_date(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = CoordinatorAgent()
        meta = agent._extract_metadata("2024年6月15日有什么大事")
        assert meta["date"] == "2024-06-15"

    def test_extract_metadata_stock(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = CoordinatorAgent()
        meta = agent._extract_metadata("茅台最近走势")
        assert meta.get("stock_code") == "600519.SH"
        assert meta.get("stock_name") == "贵州茅台"


# ===================== KLineAgent can_handle =====================


class TestKLineAgentCanHandle:

    def test_intent_kline(self):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = KLineAgent()
        ctx = AgentContext(raw_input="test", metadata={"intent": "kline"})
        assert agent.can_handle(ctx) is True

    def test_ts_code_in_metadata(self):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = KLineAgent()
        ctx = AgentContext(raw_input="test", metadata={"ts_code": "600519.SH"})
        assert agent.can_handle(ctx) is True

    def test_kline_keyword(self):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = KLineAgent()
        ctx = AgentContext(raw_input="帮我分析MACD走势")
        assert agent.can_handle(ctx) is True

    def test_stock_name(self):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = KLineAgent()
        ctx = AgentContext(raw_input="茅台最近怎么样")
        assert agent.can_handle(ctx) is True

    def test_irrelevant_query(self):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = KLineAgent()
        ctx = AgentContext(raw_input="今天天气不错")
        assert agent.can_handle(ctx) is False

    def test_process_no_stock(self, tools):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = _bind(KLineAgent(), tools)
        ctx = AgentContext(raw_input="帮我看看走势", metadata={"intent": "kline"})
        result = agent.run(ctx)
        # No stock code identifiable → should fail gracefully
        assert result.agent_name == "KLineAgent"

    def test_process_with_stock(self, tools):
        from financial_rag.agents.kline_agent import KLineAgent
        agent = _bind(KLineAgent(), tools)
        ctx = AgentContext(raw_input="分析茅台", metadata={"ts_code": "600519.SH", "name": "贵州茅台"})
        result = agent.run(ctx)
        # Will call analyze_kline tool (may fail without tushare token, but shouldn't crash)
        assert result.agent_name == "KLineAgent"


# ===================== EventImpactAgent can_handle =====================


class TestEventImpactAgentCanHandle:

    def test_intent_event_impact(self):
        from financial_rag.agents.event_impact_agent import EventImpactAgent
        agent = EventImpactAgent()
        ctx = AgentContext(raw_input="test", metadata={"intent": "event_impact"})
        assert agent.can_handle(ctx) is True

    def test_date_present(self):
        from financial_rag.agents.event_impact_agent import EventImpactAgent
        agent = EventImpactAgent()
        ctx = AgentContext(raw_input="2024-06-01发生了什么")
        assert agent.can_handle(ctx) is True

    def test_event_keyword(self):
        from financial_rag.agents.event_impact_agent import EventImpactAgent
        agent = EventImpactAgent()
        ctx = AgentContext(raw_input="这个消息是利好还是利空")
        assert agent.can_handle(ctx) is True

    def test_irrelevant(self):
        from financial_rag.agents.event_impact_agent import EventImpactAgent
        agent = EventImpactAgent()
        ctx = AgentContext(raw_input="请帮我写一封邮件")
        assert agent.can_handle(ctx) is False

    def test_process_no_date_no_keyword(self, tools):
        from financial_rag.agents.event_impact_agent import EventImpactAgent
        agent = _bind(EventImpactAgent(), tools)
        ctx = AgentContext(raw_input="分析一下影响", metadata={"intent": "event_impact"})
        result = agent.run(ctx)
        # Should fail gracefully asking for date or keyword
        assert not result.success
        assert "missing_params" in str(result.data)


# ===================== ScoringAgent can_handle =====================


class TestScoringAgentCanHandle:

    def test_has_final_answer(self):
        from financial_rag.agents.scoring_agent import ScoringAgent
        agent = ScoringAgent()
        ctx = AgentContext(raw_input="test", final_answer="some answer")
        assert agent.can_handle(ctx) is True

    def test_has_findings(self):
        from financial_rag.agents.scoring_agent import ScoringAgent
        agent = ScoringAgent()
        ctx = AgentContext(raw_input="test", intermediate_findings=[{"stage": "kline"}])
        assert agent.can_handle(ctx) is True

    def test_no_data(self):
        from financial_rag.agents.scoring_agent import ScoringAgent
        agent = ScoringAgent()
        ctx = AgentContext(raw_input="test")
        assert agent.can_handle(ctx) is False

    def test_process_scoring(self, tools):
        from financial_rag.agents.scoring_agent import ScoringAgent
        agent = _bind(ScoringAgent(), tools)
        ctx = AgentContext(
            raw_input="test query",
            final_answer="这是一份分析报告",
            intermediate_findings=[
                {"stage": "kline_analysis", "ts_code": "600519.SH"},
                {"stage": "report", "source_count": 3},
            ],
            metadata={
                "fetched_data": [{"x": 1}],
                "retrieved_items": [{"text": "doc1"}],
            },
        )
        result = agent.run(ctx)
        assert result.success
        assert result.agent_name == "ScoringAgent"
        assert "pipeline_scores" in result.data
        assert "report" in result.data


# ===================== ReportAgent can_handle =====================


class TestReportAgentCanHandle:

    def test_has_parsed_data(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        ctx = AgentContext(raw_input="test", parsed_data=[{"text": "doc"}])
        assert agent.can_handle(ctx) is True

    def test_has_intermediate_findings(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        ctx = AgentContext(raw_input="test", intermediate_findings=[{"stage": "kline"}])
        assert agent.can_handle(ctx) is True

    def test_has_final_answer(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        ctx = AgentContext(raw_input="test", final_answer="answer")
        assert agent.can_handle(ctx) is True

    def test_no_data(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        ctx = AgentContext(raw_input="test")
        assert agent.can_handle(ctx) is False

    def test_findings_to_documents_kline(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        findings = [{
            "stage": "kline_analysis",
            "ts_code": "600519.SH",
            "data_points": 60,
            "stats": {"latest_close": 1800, "period_change_pct": 5.2},
            "indicators": {"macd": {"signal": "bullish"}, "rsi": {"value": 65}},
        }]
        docs = agent._findings_to_documents(findings, "分析报告")
        assert len(docs) >= 1
        assert "KLineAgent" in docs[0]["metadata"]["source"]

    def test_findings_to_documents_event(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        findings = [{
            "stage": "event_impact",
            "event_count": 5,
            "assessment": {"overall_label": "利好", "overall_factor": 7, "summary": "积极影响"},
        }]
        docs = agent._findings_to_documents(findings, None)
        assert len(docs) >= 1
        assert "EventImpactAgent" in docs[0]["metadata"]["source"]

    def test_findings_with_final_answer(self):
        from financial_rag.agents.report_agent import ReportAgent
        agent = ReportAgent()
        findings = [{"stage": "kline_analysis", "ts_code": "600519.SH"}]
        docs = agent._findings_to_documents(findings, "综合结论文本")
        # Should include both the finding and the final answer as a doc
        assert len(docs) == 2
        assert docs[-1]["metadata"]["source"] == "specialist_summary"
