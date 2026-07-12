"""
Test core/orchestrator.py — register, execute (3 modes), retry, MessageBus lineage

Uses simple stub agents. test_orchestrator_merge.py covers _apply_updates separately.
"""
import pytest
from unittest.mock import patch

from financial_rag.core.base import (
    BaseAgent, AgentContext, AgentResult, ExecutionMode, AgentStatus,
)
from financial_rag.core.orchestrator import (
    AgentOrchestrator, CoordinatorConfig, ExecutionResult,
)
from financial_rag.core.protocol import MessageBus


# ===================== Stub Agents =====================


class StubAgent(BaseAgent):
    """Configurable stub agent"""

    def __init__(self, name, success=True, updates=None, can_handle_val=True):
        super().__init__(name=name, description=f"Stub {name}")
        self._success = success
        self._updates = updates or {}
        self._can = can_handle_val
        self.call_count = 0

    def process(self, context: AgentContext) -> AgentResult:
        self.call_count += 1
        return AgentResult(
            success=self._success,
            message=f"{self.name} {'ok' if self._success else 'fail'}",
            agent_name=self.name,
            context_updates=self._updates,
        )

    def can_handle(self, context: AgentContext) -> bool:
        return self._can


class FailThenSucceedAgent(BaseAgent):
    """Fails N times then succeeds"""

    def __init__(self, name, fail_times=1):
        super().__init__(name=name, description="flaky")
        self._fail_times = fail_times
        self.call_count = 0

    def process(self, context):
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise RuntimeError("transient error")
        return AgentResult(success=True, message="recovered", agent_name=self.name)


class AlwaysFailAgent(BaseAgent):
    def __init__(self, name="FailBot"):
        super().__init__(name=name, description="always fails")
        self.call_count = 0

    def process(self, context):
        self.call_count += 1
        raise RuntimeError("permanent failure")


# ===================== Fixtures =====================


@pytest.fixture
def orch():
    return AgentOrchestrator(CoordinatorConfig(verbose=False))


@pytest.fixture
def ctx():
    return AgentContext(raw_input="test query", metadata={"intent": "general"})


# ===================== Registration =====================


class TestRegistration:

    def test_register_adds_agent(self, orch):
        agent = StubAgent("A")
        orch.register(agent)
        assert "A" in orch.agents
        assert "A" in orch.pipeline

    def test_register_with_position(self, orch):
        orch.register(StubAgent("A"))
        orch.register(StubAgent("B"))
        orch.register(StubAgent("C"), position=1)
        assert orch.pipeline == ["A", "C", "B"]

    def test_register_all(self, orch):
        orch.register_all(StubAgent("A"), StubAgent("B"), StubAgent("C"))
        assert len(orch.agents) == 3
        assert orch.pipeline == ["A", "B", "C"]

    def test_set_pipeline_reorder(self, orch):
        orch.register_all(StubAgent("A"), StubAgent("B"), StubAgent("C"))
        orch.set_pipeline(["C", "A", "B"])
        assert orch.pipeline == ["C", "A", "B"]

    def test_set_pipeline_unknown_raises(self, orch):
        orch.register(StubAgent("A"))
        with pytest.raises(ValueError, match="未注册"):
            orch.set_pipeline(["A", "Unknown"])


# ===================== Sequential Execution =====================


class TestSequentialExecution:

    def test_basic_sequential(self, orch, ctx):
        orch.config.execution_mode = ExecutionMode.SEQUENTIAL
        a1 = StubAgent("A1", updates={"final_answer": "result1"})
        a2 = StubAgent("A2", updates={"final_answer": "result2"})
        orch.register_all(a1, a2)

        result = orch.execute("test", context=ctx)
        assert result.success is True
        assert len(result.agent_results) == 2
        assert a1.call_count == 1
        assert a2.call_count == 1
        # Last agent's answer wins
        assert result.final_output == "result2"

    def test_sequential_skips_can_not_handle(self, orch, ctx):
        orch.config.execution_mode = ExecutionMode.SEQUENTIAL
        a1 = StubAgent("A1", can_handle_val=False)
        a2 = StubAgent("A2")
        orch.register_all(a1, a2)

        result = orch.execute("test", context=ctx)
        assert a1.call_count == 0
        assert a2.call_count == 1

    def test_sequential_continues_on_failure(self, orch, ctx):
        orch.config.execution_mode = ExecutionMode.SEQUENTIAL
        a1 = StubAgent("A1", success=False)
        a2 = StubAgent("A2")
        orch.register_all(a1, a2)

        result = orch.execute("test", context=ctx)
        # a1 failed, but chain continues
        assert a2.call_count == 1
        # Overall success = False because a1 failed
        assert result.success is False


# ===================== Parallel Execution =====================


class TestParallelExecution:

    def test_basic_parallel(self, orch, ctx):
        orch.config.execution_mode = ExecutionMode.PARALLEL
        orch.config.max_parallel_agents = 3
        a1 = StubAgent("A1")
        a2 = StubAgent("A2")
        orch.register_all(a1, a2)

        result = orch.execute("test", context=ctx)
        assert result.success is True
        assert len(result.agent_results) == 2

    def test_parallel_handles_timeout(self, orch, ctx):
        orch.config.execution_mode = ExecutionMode.PARALLEL
        orch.config.timeout_seconds = 0.001  # very short timeout
        orch.config.max_parallel_agents = 2

        class SlowAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="Slow", description="slow")

            def process(self, context):
                import time; time.sleep(2)
                return AgentResult(success=True, message="done", agent_name="Slow")

        orch.register(SlowAgent())
        result = orch.execute("test", context=ctx)
        # Should have a failure result due to timeout
        assert len(result.agent_results) >= 1


# ===================== Conditional Execution =====================


class TestConditionalExecution:

    def test_conditional_continues_on_failure(self, orch, ctx):
        orch.config.execution_mode = ExecutionMode.CONDITIONAL
        a1 = StubAgent("A1", success=False)
        a2 = StubAgent("A2")
        orch.register_all(a1, a2)

        result = orch.execute("test", context=ctx)
        assert a1.call_count == 1
        assert a2.call_count == 1  # continues despite a1 failure


# ===================== Retry Behavior =====================


class TestRetry:
    """Retry behavior: _run_one retries on exception from agent.run().
    
    Note: BaseAgent.run() catches exceptions from process() and returns
    AgentResult(success=False). So retries in _run_one only trigger when
    run() itself raises (e.g. agent.run is monkeypatched to raise).
    """

    def test_agent_failure_propagates_as_failed_result(self, orch, ctx):
        """process() raises → run() catches → returns failed AgentResult"""
        orch.config.max_retries = 2
        agent = AlwaysFailAgent("FailBot")
        orch.register(agent)

        result = orch.execute("test", context=ctx)
        assert result.success is False
        # run() catches the exception and returns success=False
        assert agent.call_count >= 1

    def test_flaky_agent_returns_failure_not_retried(self, orch, ctx):
        """BaseAgent.run() wraps exceptions, so _run_one sees success=False not exception.
        The retry loop only catches exceptions, not failed results."""
        orch.config.max_retries = 2
        agent = FailThenSucceedAgent("Flaky", fail_times=1)
        orch.register(agent)

        result = orch.execute("test", context=ctx)
        # First call: process() raises → run() returns AgentResult(success=False)
        # _run_one sees success=False but no exception → no retry
        # Result is failed
        assert result.success is False
        assert agent.call_count == 1


# ===================== Execution Result =====================


class TestExecutionResult:

    def test_get_agent_result(self):
        r1 = AgentResult(success=True, agent_name="A", message="a done")
        r2 = AgentResult(success=True, agent_name="B", message="b done")
        er = ExecutionResult(success=True, agent_results=[r1, r2])
        assert er.get("A").message == "a done"
        assert er.get("B").message == "b done"
        assert er.get("C") is None

    def test_execution_time_nonnegative(self, orch, ctx):
        orch.register(StubAgent("A"))
        result = orch.execute("test", context=ctx)
        assert result.execution_time >= 0

    def test_history_appended(self, orch, ctx):
        orch.register(StubAgent("A"))
        orch.execute("test1", context=ctx)
        orch.execute("test2")
        assert len(orch.history) == 2


# ===================== MessageBus Integration =====================


class TestMessageBusIntegration:

    def test_apply_updates_publishes_to_bus(self, orch, ctx):
        bus = MessageBus()
        orch.message_bus = bus
        orch.use_message_bus = True
        orch.context = ctx

        r = AgentResult(
            success=True, agent_name="TestAgent",
            context_updates={"final_answer": "hello"},
        )
        orch._apply_updates(r)

        assert len(bus) > 0
        # Should have data + done messages
        all_msgs = bus.consume("all")
        types = {m.msg_type for m in all_msgs}
        assert "data" in types
        assert "done" in types

    def test_get_data_lineage_with_bus(self, orch, ctx):
        bus = MessageBus()
        orch.message_bus = bus
        orch.use_message_bus = True
        orch.register(StubAgent("A", updates={"final_answer": "test"}))

        orch.execute("test", context=ctx)
        lineage = orch.get_data_lineage()
        assert len(lineage) > 0
        assert all("msg_id" in item for item in lineage)
        assert all("sender" in item for item in lineage)

    def test_get_data_lineage_without_bus(self, orch):
        assert orch.get_data_lineage() == []


# ===================== Reset =====================


class TestReset:

    def test_reset_clears_context(self, orch, ctx):
        orch.register(StubAgent("A"))
        orch.execute("test", context=ctx)
        assert orch.context is not None

        orch.reset()
        assert orch.context is None

    def test_execute_creates_context_from_raw_input(self, orch):
        orch.register(StubAgent("A"))
        result = orch.execute("hello world")
        assert orch.context.raw_input == "hello world"
