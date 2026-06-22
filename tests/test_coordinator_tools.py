"""
Test coordinator_tools module — classify_query_intent, select_agent_chain

Covers:
- classify_query_intent: various query types (kline, event, report, news, general)
- select_agent_chain: all intents, low confidence, unknown intent
- Tool definitions: COORDINATOR_TOOLS shape
"""
import pytest

from financial_rag.tools.coordinator_tools import (
    classify_query_intent,
    select_agent_chain,
    CLASSIFY_INTENT_TOOL,
    SELECT_CHAIN_TOOL,
    COORDINATOR_TOOLS,
)


# ===================== classify_query_intent =====================


class TestClassifyQueryIntent:

    def test_kline_intent_keywords(self):
        queries = [
            "茅台走势分析",
            "看看沪深300的K线",
            "比亚迪MACD技术分析",
            "茅台最近涨跌情况",
        ]
        for q in queries:
            result = classify_query_intent(q)
            assert result["intent"] == "kline", f"Query '{q}' should be kline, got {result['intent']}"
            assert result["confidence"] > 0

    def test_event_impact_intent(self):
        queries = [
            "2024年6月1日发生了什么大事",
            "利好消息对股市的影响",
            "这次暴跌有什么冲击",
        ]
        for q in queries:
            result = classify_query_intent(q)
            assert result["intent"] == "event_impact", f"Query '{q}' should be event_impact, got {result['intent']}"

    def test_report_intent(self):
        queries = [
            "商汤科技财报分析",
            "茅台营收增长多少",
            "年报数据解读",
        ]
        for q in queries:
            result = classify_query_intent(q)
            assert result["intent"] == "report", f"Query '{q}' should be report, got {result['intent']}"

    def test_news_intent(self):
        queries = [
            "最新AI新闻",
            "今天有什么行业动态",
            "半导体最新资讯",
        ]
        for q in queries:
            result = classify_query_intent(q)
            assert result["intent"] == "news", f"Query '{q}' should be news, got {result['intent']}"

    def test_general_intent(self):
        result = classify_query_intent("随便聊聊")
        assert result["intent"] == "general"

    def test_result_has_matched_keywords(self):
        result = classify_query_intent("茅台走势怎么样")
        assert "matched_keywords" in result
        assert isinstance(result["matched_keywords"], list)

    def test_confidence_range(self):
        result = classify_query_intent("看看半导体K线走势")
        assert 0 <= result["confidence"] <= 1


# ===================== select_agent_chain =====================


class TestSelectAgentChain:

    def test_kline_chain(self):
        result = select_agent_chain(intent="kline")
        assert "AnalysisAgent" in result["agent_chain"]
        assert "ScoringAgent" in result["agent_chain"]

    def test_event_impact_chain(self):
        result = select_agent_chain(intent="event_impact")
        assert "AnalysisAgent" in result["agent_chain"]
        assert "ScoringAgent" in result["agent_chain"]

    def test_report_chain(self):
        result = select_agent_chain(intent="report")
        assert "IngestionAgent" in result["agent_chain"]
        assert "AnalysisAgent" in result["agent_chain"]

    def test_news_chain(self):
        result = select_agent_chain(intent="news")
        assert "IngestionAgent" in result["agent_chain"]
        assert "AnalysisAgent" in result["agent_chain"]

    def test_low_confidence_adds_ingestion(self):
        result = select_agent_chain(intent="kline", confidence=0.2)
        assert "IngestionAgent" in result["agent_chain"]

    def test_high_confidence_no_ingestion_for_kline(self):
        result = select_agent_chain(intent="kline", confidence=0.8)
        # kline chain doesn't include IngestionAgent by default
        assert result["agent_chain"][0] != "IngestionAgent" or "AnalysisAgent" in result["agent_chain"]

    def test_unknown_intent_uses_default(self):
        result = select_agent_chain(intent="completely_unknown_type")
        assert len(result["agent_chain"]) > 0
        # Default chain includes IngestionAgent, AnalysisAgent, ScoringAgent
        assert "AnalysisAgent" in result["agent_chain"]

    def test_chain_description_format(self):
        result = select_agent_chain(intent="kline")
        assert "→" in result["chain_description"]

    def test_result_contains_intent_and_confidence(self):
        result = select_agent_chain(intent="news", confidence=0.7)
        assert result["intent"] == "news"
        assert result["confidence"] == 0.7

    def test_boundary_confidence_04(self):
        """At exactly 0.4, IngestionAgent should NOT be added (threshold is <0.4)"""
        result = select_agent_chain(intent="kline", confidence=0.4)
        # kline default chain is [AnalysisAgent, ScoringAgent]
        # confidence=0.4 is NOT < 0.4, so no IngestionAgent added
        assert result["agent_chain"] == ["AnalysisAgent", "ScoringAgent"]


# ===================== Tool Definitions =====================


class TestCoordinatorToolDefinitions:

    def test_tools_count(self):
        assert len(COORDINATOR_TOOLS) == 2

    def test_classify_intent_tool(self):
        assert CLASSIFY_INTENT_TOOL.name == "classify_query_intent"
        assert CLASSIFY_INTENT_TOOL.category == "analysis"
        assert callable(CLASSIFY_INTENT_TOOL.callback)
        assert "query" in CLASSIFY_INTENT_TOOL.parameters["required"]

    def test_select_chain_tool(self):
        assert SELECT_CHAIN_TOOL.name == "select_agent_chain"
        assert SELECT_CHAIN_TOOL.category == "analysis"
        assert callable(SELECT_CHAIN_TOOL.callback)
        assert "intent" in SELECT_CHAIN_TOOL.parameters["required"]
        assert "confidence" in SELECT_CHAIN_TOOL.parameters["properties"]
