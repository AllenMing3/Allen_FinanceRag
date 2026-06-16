"""
Test agents — can_handle + process with tool delegation

Covers:
- CoordinatorAgent: can_handle (always True), process (intent classification + chain selection)
- AnalysisAgent: can_handle (always True), intent-based dispatch, render helpers
- ScoringAgent: can_handle (findings/final_answer), process (scoring pipeline)
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
        assert "AnalysisAgent" in data["agent_chain"]

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


# ===================== AnalysisAgent =====================


class TestAnalysisAgent:

    def test_can_handle_always_true(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        ctx = AgentContext(raw_input="test")
        assert agent.can_handle(ctx) is True

    def test_kline_no_stock(self, tools):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = _bind(AnalysisAgent(), tools)
        ctx = AgentContext(raw_input="帮我看看走势", metadata={"intent": "kline"})
        result = agent.run(ctx)
        assert result.agent_name == "AnalysisAgent"

    def test_kline_with_stock(self, tools):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = _bind(AnalysisAgent(), tools)
        ctx = AgentContext(raw_input="分析茅台", metadata={
            "intent": "kline", "ts_code": "600519.SH", "stock_name": "贵州茅台",
        })
        result = agent.run(ctx)
        assert result.agent_name == "AnalysisAgent"

    def test_event_no_params(self, tools):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = _bind(AnalysisAgent(), tools)
        ctx = AgentContext(raw_input="分析一下", metadata={"intent": "event_impact"})
        result = agent.run(ctx)
        assert result.agent_name == "AnalysisAgent"

    def test_extraction_with_docs(self, tools):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = _bind(AnalysisAgent(), tools)
        docs = [{"text": "商汤科技2024年营收50.3亿元，同比增长36%", "meta": {"source": "test"}}]
        ctx = AgentContext(
            parsed_data=docs, raw_input="分析财报",
            metadata={"intent": "report"},
        )
        result = agent.run(ctx)
        assert result.success
        assert result.agent_name == "AnalysisAgent"

    def test_render_markdown(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        report = {
            "title": "Test Report",
            "key_findings": [{"finding": "test finding", "importance": "high", "source_refs": [1]}],
            "summary": "test summary",
        }
        sources = [{"id": 1, "title": "Source 1", "source": "test", "date": "2024-01-01"}]
        md = agent._render_markdown(report, sources)
        assert "# Test Report" in md
        assert "test finding" in md
        assert "[1]" in md

    def test_findings_to_documents(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        findings = [
            {"stage": "kline_analysis", "ts_code": "600519.SH", "data_points": 60},
            {"stage": "event_impact", "event_count": 3},
        ]
        docs = agent._findings_to_documents(findings, "综合结论")
        assert len(docs) == 3  # 2 findings + 1 final answer
        assert docs[-1]["meta"]["source"] == "specialist_summary"

    def test_build_sources(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        docs = [
            {"text": "hello", "meta": {"source": "news", "title": "Title1"}},
            {"text": "world", "meta": {"source": "report"}},
        ]
        sources = agent._build_sources(docs)
        assert len(sources) == 2
        assert sources[0]["source"] == "news"
        assert sources[1]["title"] == "world"[:60]

    def test_evaluate_extraction(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        metrics = {"revenue": 100, "net_income": 50}
        entities = {"companies": ["A"], "ai_models": ["B"]}
        score = agent._evaluate_extraction(metrics, entities)
        assert 0 <= score <= 1

    def test_evaluate_queries(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        assert agent._evaluate_queries([]) == 0.0
        assert agent._evaluate_queries(["q1", "q2", "q3"]) > 0

    def test_extract_stock_code(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        ts_code, name = agent._extract_stock_code("茅台走势")
        assert ts_code == "600519.SH"
        ts_code2, _ = agent._extract_stock_code("600036走势")
        assert ts_code2 == "600036.SH"

    def test_extract_date(self):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        assert agent._extract_date("2024年6月15日") == "2024-06-15"
        assert agent._extract_date("2024-03-01") == "2024-03-01"
        assert agent._extract_date("no date here") == ""


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
