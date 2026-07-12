"""
Test retrievers/query_planner.py — QueryPlanner + QueryPlan + SubQuery

Tests plan(), _parse_plan(), _fallback() with mock LLMCaller.
"""
import pytest
from unittest.mock import MagicMock

from financial_rag.retrievers.query_planner import (
    QueryPlanner, QueryPlan, SubQuery,
)


# ===================== QueryPlan / SubQuery data classes =====================


class TestQueryPlan:

    def test_is_simple_single_subquery(self):
        plan = QueryPlan(
            original_query="test",
            sub_queries=[SubQuery(query="q1")],
        )
        assert plan.is_simple is True

    def test_is_simple_empty_subqueries(self):
        plan = QueryPlan(original_query="test", sub_queries=[])
        assert plan.is_simple is True

    def test_is_simple_multiple_subqueries(self):
        plan = QueryPlan(
            original_query="test",
            sub_queries=[SubQuery(query="q1"), SubQuery(query="q2")],
        )
        assert plan.is_simple is False

    def test_defaults(self):
        plan = QueryPlan(original_query="hello")
        assert plan.intent == "factual"
        assert plan.strategy == "parallel"
        assert plan.sub_queries == []


class TestSubQuery:

    def test_defaults(self):
        sq = SubQuery(query="test query")
        assert sq.source == "all"
        assert sq.mode == "hybrid"
        assert sq.purpose == ""

    def test_custom_values(self):
        sq = SubQuery(query="q", source="kb", mode="local", purpose="lookup")
        assert sq.source == "kb"
        assert sq.mode == "local"
        assert sq.purpose == "lookup"


# ===================== QueryPlanner =====================


class TestQueryPlanner:

    def _make_planner(self, mock_response=None):
        caller = MagicMock()
        caller.call_json.return_value = mock_response
        return QueryPlanner(caller=caller), caller

    def test_plan_success_comparison(self):
        mock_resp = {
            "intent": "comparison",
            "strategy": "parallel",
            "sub_queries": [
                {"query": "英伟达芯片参数", "source": "all", "mode": "local", "purpose": "获取NV数据"},
                {"query": "华为芯片参数", "source": "all", "mode": "local", "purpose": "获取华为数据"},
                {"query": "芯片对比分析", "source": "all", "mode": "hybrid", "purpose": "对比"},
            ],
        }
        planner, caller = self._make_planner(mock_resp)
        plan = planner.plan("英伟达和华为AI芯片对比")

        assert plan.intent == "comparison"
        assert plan.strategy == "parallel"
        assert len(plan.sub_queries) == 3
        assert plan.sub_queries[0].query == "英伟达芯片参数"
        assert plan.sub_queries[2].mode == "hybrid"
        assert plan.is_simple is False

        caller.call_json.assert_called_once()

    def test_plan_success_simple_factual(self):
        mock_resp = {
            "intent": "factual",
            "strategy": "parallel",
            "sub_queries": [
                {"query": "茅台收盘价", "source": "kb", "mode": "local", "purpose": "查询"},
            ],
        }
        planner, _ = self._make_planner(mock_resp)
        plan = planner.plan("茅台今天收盘价多少")
        assert plan.intent == "factual"
        assert plan.is_simple is True

    def test_plan_llm_returns_none_fallback(self):
        planner, _ = self._make_planner(None)
        plan = planner.plan("test query")
        assert plan.intent == "factual"
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].query == "test query"
        assert plan.sub_queries[0].source == "all"

    def test_plan_llm_returns_string_fallback(self):
        planner, _ = self._make_planner("not a dict")
        plan = planner.plan("test")
        assert plan.is_simple is True

    def test_plan_llm_returns_empty_subqueries(self):
        mock_resp = {"intent": "summary", "sub_queries": []}
        planner, _ = self._make_planner(mock_resp)
        plan = planner.plan("AI行业最近怎么样")
        assert plan.intent == "summary"
        assert len(plan.sub_queries) == 0
        assert plan.is_simple is True

    def test_plan_llm_raises_propagates(self):
        """plan() does not wrap call_json exceptions — they propagate"""
        caller = MagicMock()
        caller.call_json.side_effect = RuntimeError("API error")
        planner = QueryPlanner(caller=caller)
        with pytest.raises(RuntimeError, match="API error"):
            planner.plan("test")

    def test_parse_plan_with_defaults(self):
        planner, _ = self._make_planner(None)
        raw = {
            "intent": "deep_dive",
            "sub_queries": [
                {"query": "q1"},  # missing source/mode/purpose
            ],
        }
        plan = planner._parse_plan("test", raw)
        assert plan.intent == "deep_dive"
        assert plan.sub_queries[0].source == "all"
        assert plan.sub_queries[0].mode == "mix"
        assert plan.sub_queries[0].purpose == ""

    def test_parse_plan_skips_invalid_subqueries(self):
        planner, _ = self._make_planner(None)
        raw = {
            "sub_queries": [
                {"query": "valid"},
                {"no_query_key": "skip this"},
                {"query": ""},  # empty query
                "not a dict",
            ],
        }
        plan = planner._parse_plan("test", raw)
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].query == "valid"

    def test_fallback(self):
        planner, _ = self._make_planner(None)
        plan = planner._fallback("some query")
        assert plan.original_query == "some query"
        assert plan.intent == "factual"
        assert plan.strategy == "parallel"
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].query == "some query"
        assert plan.sub_queries[0].source == "all"
        assert plan.sub_queries[0].mode == "mix"
        assert plan.sub_queries[0].purpose == "直接检索"
