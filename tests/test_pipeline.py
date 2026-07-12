"""
Test core/pipeline.py — PipelineScheduler, PipelineConfig, PipelineResult

Tests phase control, routing, and create_pipeline_scheduler factory.
Heavy mocking for external dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from financial_rag.core.pipeline import (
    PipelineScheduler, PipelineConfig, PipelineResult,
    create_pipeline_scheduler,
)
from financial_rag.core.orchestrator import AgentOrchestrator, ExecutionResult, CoordinatorConfig
from financial_rag.core.base import AgentResult, AgentContext


# ===================== Fixtures =====================


@pytest.fixture
def mock_orch():
    orch = MagicMock(spec=AgentOrchestrator)
    orch.agents = {"IngestionAgent": MagicMock(), "AnalysisAgent": MagicMock()}
    orch.pipeline = ["IngestionAgent", "AnalysisAgent"]
    orch.agent_router = None
    orch.set_pipeline = MagicMock(return_value=orch)
    orch.execute = MagicMock(return_value=ExecutionResult(
        success=True,
        agent_results=[AgentResult(success=True, agent_name="AnalysisAgent", message="ok")],
        final_output="",
    ))
    return orch


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.documents = []
    r.search = MagicMock(return_value=[
        {"text": "doc1 content", "score": 0.9, "retriever": "bm25"},
        {"text": "doc2 content", "score": 0.8, "retriever": "vector"},
    ])
    r.index = MagicMock()
    r.add = MagicMock()
    return r


@pytest.fixture
def scheduler_all_disabled(mock_orch):
    """All phases disabled — should return immediately"""
    return PipelineScheduler(
        orchestrator=mock_orch,
        config=PipelineConfig(
            enable_data_fetch=False,
            enable_index=False,
            enable_agent_analysis=False,
            enable_slot_output=False,
            enable_scoring=False,
            verbose=False,
        ),
    )


# ===================== PipelineConfig =====================


class TestPipelineConfig:

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.enable_data_fetch is True
        assert cfg.enable_index is True
        assert cfg.enable_agent_analysis is True
        assert cfg.enable_slot_output is True
        assert cfg.enable_scoring is True
        assert cfg.verbose is True


# ===================== PipelineResult =====================


class TestPipelineResult:

    def test_defaults(self):
        r = PipelineResult()
        assert r.query == ""
        assert r.success is False
        assert r.fetched_data == []
        assert r.retrieved_items == []
        assert r.errors == []
        assert r.final_output == ""
        assert r.total_elapsed_ms == 0.0

    def test_with_query(self):
        r = PipelineResult(query="test query")
        assert r.query == "test query"


# ===================== Run with all phases disabled =====================


class TestRunAllDisabled:

    def test_run_all_disabled_returns_success(self, scheduler_all_disabled):
        result = scheduler_all_disabled.run("test query")
        assert result.success is True
        assert result.query == "test query"
        assert result.total_elapsed_ms >= 0
        assert result.errors == []

    def test_run_all_disabled_no_fetch(self, scheduler_all_disabled):
        result = scheduler_all_disabled.run("test")
        assert result.fetched_data == []
        assert result.fetch_elapsed_ms == 0

    def test_run_all_disabled_no_index(self, scheduler_all_disabled):
        result = scheduler_all_disabled.run("test")
        assert result.retrieved_items == []
        assert result.indexed_docs == 0


# ===================== Phase 1: Fetch =====================


class TestPhaseFetch:

    def test_fetch_skip_when_disabled(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            config=PipelineConfig(enable_data_fetch=False, verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_fetch("test", result, 10)
        assert out.fetched_data == []

    def test_fetch_skip_no_llm(self, mock_orch, mock_retriever):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            retriever=mock_retriever,
            llm=None,  # No LLM
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_fetch("test", result, 10)
        assert out.fetched_data == []

    def test_fetch_skip_no_registry(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            registry=None,
            llm=MagicMock(),
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_fetch("test", result, 10)
        assert out.fetched_data == []


# ===================== Phase 2: Index =====================


class TestPhaseIndex:

    def test_index_skip_when_disabled(self, mock_orch, mock_retriever):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            retriever=mock_retriever,
            config=PipelineConfig(enable_index=False, verbose=False),
        )
        result = PipelineResult(query="test", fetched_data=[{"title": "t", "content": "c"}])
        out = sched._phase_index("test", result, 5)
        mock_retriever.index.assert_not_called()
        assert out.retrieved_items == []

    def test_index_with_data(self, mock_orch, mock_retriever):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            retriever=mock_retriever,
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(
            query="test",
            fetched_data=[
                {"title": "title1", "content": "content1", "source": "rss", "publish_time": "", "url": ""},
            ],
        )
        out = sched._phase_index("test query", result, 5)
        assert out.indexed_docs >= 1
        assert len(out.retrieved_items) == 2  # mock returns 2

    def test_index_no_data_no_retriever(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            retriever=None,
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_index("test", result, 5)
        assert out.retrieved_items == []


# ===================== Phase 3: Process =====================


class TestPhaseProcess:

    def test_process_skip_when_disabled(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            config=PipelineConfig(enable_agent_analysis=False, verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_process("test", result)
        mock_orch.execute.assert_not_called()

    @patch('financial_rag.core.agent_router.create_agent_router')
    def test_process_calls_orchestrator(self, mock_create_router, mock_orch):
        mock_router = MagicMock()
        mock_decision = MagicMock()
        mock_decision.intent = "kline"
        mock_decision.agent_chain = ["AnalysisAgent"]
        mock_decision.metadata = {}
        mock_router.route.return_value = mock_decision
        mock_create_router.return_value = mock_router

        sched = PipelineScheduler(
            orchestrator=mock_orch,
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(
            query="茅台走势",
            retrieved_items=[{"text": "茅台今天涨了2%"}],
        )
        out = sched._phase_process("茅台走势", result)
        mock_orch.execute.assert_called_once()
        assert out.agent_exec_result is not None


# ===================== Phase 4: Output =====================


class TestPhaseOutput:

    def test_output_skip_when_disabled(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            config=PipelineConfig(enable_slot_output=False, verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_output("test", result)
        assert out.final_output == ""

    def test_output_skip_when_agent_already_produced(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            filler=MagicMock(),
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(query="test", final_output="x" * 100)
        out = sched._phase_output("test", result)
        assert out.final_output == "x" * 100  # unchanged

    def test_output_skip_no_filler(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            filler=None,
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_output("test", result)
        assert out.final_output == ""

    def test_output_runs_filler(self, mock_orch):
        mock_filler = MagicMock()
        mock_fill_stats = MagicMock()
        mock_fill_stats.filled_slots = 3
        mock_fill_stats.total_slots = 5
        mock_fill_stats.avg_ttft_ms = 100
        mock_fill_stats.template_name = "quick_qa"
        mock_filler.fill.return_value = mock_fill_stats
        mock_filler.render.return_value = "rendered output"

        sched = PipelineScheduler(
            orchestrator=mock_orch,
            filler=mock_filler,
            config=PipelineConfig(verbose=False),
        )
        result = PipelineResult(query="test query")
        out = sched._phase_output("test query", result)
        assert out.final_output == "rendered output"
        assert out.fill_stats == mock_fill_stats
        mock_filler.fill.assert_called_once()


# ===================== Phase 5: Evolve =====================


class TestPhaseEvolve:

    def test_evolve_skip_when_disabled(self, mock_orch):
        sched = PipelineScheduler(
            orchestrator=mock_orch,
            config=PipelineConfig(enable_scoring=False, verbose=False),
        )
        result = PipelineResult(query="test")
        out = sched._phase_evolve(result)
        assert out.scorecard is None


# ===================== _route_query =====================


class TestRouteQuery:

    @patch('financial_rag.core.agent_router.create_agent_router')
    def test_route_query_uses_existing_router(self, mock_create_router, mock_orch):
        mock_router = MagicMock()
        mock_decision = MagicMock()
        mock_decision.intent = "news"
        mock_decision.agent_chain = ["IngestionAgent", "AnalysisAgent"]
        mock_decision.metadata = {}
        mock_router.route.return_value = mock_decision
        mock_orch.agent_router = mock_router

        sched = PipelineScheduler(orchestrator=mock_orch, config=PipelineConfig(verbose=False))
        result = PipelineResult(query="test")
        routing = sched._route_query("最新AI新闻", result)

        assert routing.intent == "news"
        mock_create_router.assert_not_called()  # used existing

    @patch('financial_rag.core.agent_router.create_agent_router')
    def test_route_query_creates_router_if_missing(self, mock_create_router, mock_orch):
        mock_router = MagicMock()
        mock_decision = MagicMock()
        mock_decision.intent = "general"
        mock_decision.agent_chain = ["AnalysisAgent"]
        mock_decision.metadata = {}
        mock_router.route.return_value = mock_decision
        mock_create_router.return_value = mock_router
        mock_orch.agent_router = None

        sched = PipelineScheduler(orchestrator=mock_orch, config=PipelineConfig(verbose=False))
        result = PipelineResult(query="test")
        routing = sched._route_query("商汤科技分析", result)

        assert routing.intent == "general"
        mock_create_router.assert_called_once()

    @patch('financial_rag.core.agent_router.create_agent_router')
    def test_route_query_passes_fetched_date(self, mock_create_router, mock_orch):
        mock_router = MagicMock()
        mock_decision = MagicMock()
        mock_decision.intent = "event_impact"
        mock_decision.agent_chain = ["AnalysisAgent"]
        mock_decision.metadata = {"event_date": "2025-01-15"}
        mock_router.route.return_value = mock_decision
        mock_create_router.return_value = mock_router
        mock_orch.agent_router = None

        sched = PipelineScheduler(orchestrator=mock_orch, config=PipelineConfig(verbose=False))
        result = PipelineResult(
            query="降准影响",
            fetched_data=[{"title": "央行降准", "publish_time": "2025-01-15T10:00:00"}],
        )
        routing = sched._route_query("降准影响", result)

        # Verify route was called with context containing fetched_date
        call_args = mock_router.route.call_args
        assert call_args[1]["context"]["fetched_date"] == "2025-01-15"


# ===================== Factory =====================


class TestFactory:

    def test_create_pipeline_scheduler(self, mock_orch, mock_retriever):
        sched = create_pipeline_scheduler(
            orchestrator=mock_orch,
            retriever=mock_retriever,
            config=PipelineConfig(verbose=False),
        )
        assert isinstance(sched, PipelineScheduler)
        assert sched.orchestrator is mock_orch
        assert sched.retriever is mock_retriever
        assert sched.config.verbose is False

    def test_create_pipeline_scheduler_defaults(self, mock_orch):
        sched = create_pipeline_scheduler(orchestrator=mock_orch)
        assert sched.config.verbose is True
        assert sched.llm is None
        assert sched.filler is None


# ===================== Full Run Integration (mocked) =====================


class TestFullRun:

    def test_run_exception_handled(self, mock_orch):
        mock_orch.execute.side_effect = RuntimeError("boom")

        sched = PipelineScheduler(
            orchestrator=mock_orch,
            config=PipelineConfig(
                enable_data_fetch=False,
                enable_index=False,
                enable_slot_output=False,
                enable_scoring=False,
                verbose=False,
            ),
        )
        result = sched.run("test")
        # Exception during process phase → success=False, error captured
        assert result.success is False or len(result.errors) > 0
