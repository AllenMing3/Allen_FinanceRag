"""
Test event_impact_tools module — _fallback_assess, assess_event_impact, tool definitions

Covers:
- _fallback_assess: bullish, bearish, mixed, neutral events
- _fallback_assess: impact_factor calculation
- assess_event_impact: no-LLM fallback path
- assess_event_impact: empty events return error
- inject_event_llm: sets reference
- Tool definitions: EVENT_IMPACT_TOOLS shape
"""
import pytest

from financial_rag.tools.event_impact_tools import (
    _fallback_assess,
    assess_event_impact,
    inject_event_llm,
    _llm_ref,
    EVENT_IMPACT_TOOLS,
)


# ===================== _fallback_assess =====================


class TestFallbackAssess:

    def test_bullish_events(self):
        events = [
            {"title": "公司营收增长30%，获得重大融资突破", "content": "业绩超预期"},
            {"title": "新产品获批上市，利润创新高", "content": "股价涨停"},
        ]
        result = _fallback_assess(events)
        assert result["overall_impact"] == "bullish"
        assert result["overall_label"] == "综合利好"
        assert len(result["assessments"]) == 2
        for a in result["assessments"]:
            assert a["impact"] == "bullish"
            assert a["impact_factor"] >= 3

    def test_bearish_events(self):
        events = [
            {"title": "公司亏损严重，面临退市风险", "content": "股价暴跌"},
            {"title": "高管违规被处罚，减持预警", "content": "诉讼风险"},
        ]
        result = _fallback_assess(events)
        assert result["overall_impact"] == "bearish"
        assert result["overall_label"] == "综合利空"
        for a in result["assessments"]:
            assert a["impact"] == "bearish"

    def test_mixed_events(self):
        events = [
            {"title": "公司获得重大合同，业绩增长", "content": "超预期"},
            {"title": "公司面临退市风险，股价暴跌", "content": "亏损严重"},
        ]
        result = _fallback_assess(events)
        assert len(result["assessments"]) == 2
        # One bullish, one bearish → neutral overall
        impacts = [a["impact"] for a in result["assessments"]]
        assert "bullish" in impacts
        assert "bearish" in impacts
        assert result["overall_impact"] == "neutral"

    def test_neutral_event(self):
        events = [{"title": "公司发布公告", "content": "日常运营信息"}]
        result = _fallback_assess(events)
        assert result["assessments"][0]["impact"] == "neutral"
        assert result["assessments"][0]["impact_factor"] == 2

    def test_impact_factor_capped_at_10(self):
        # Create event with many bullish keywords
        text = "增长上涨突破利好新高超预期获批盈利扩大景气回暖涨停买入增持推荐"
        events = [{"title": text, "content": ""}]
        result = _fallback_assess(events)
        assert result["assessments"][0]["impact_factor"] <= 10

    def test_empty_events(self):
        result = _fallback_assess([])
        assert result["assessments"] == []
        assert result["overall_impact"] == "neutral"
        assert result["overall_label"] == "综合中性"
        assert result["overall_factor"] <= 10

    def test_event_title_truncated(self):
        events = [{"title": "A" * 100, "content": ""}]
        result = _fallback_assess(events)
        assert len(result["assessments"][0]["event"]) <= 40

    def test_engine_label(self):
        result = _fallback_assess([{"title": "test", "content": ""}])
        assert result["engine"] == "keyword_rules"


# ===================== assess_event_impact =====================


class TestAssessEventImpact:

    def setup_method(self):
        """Ensure no LLM is injected before each test"""
        _llm_ref["llm"] = None

    def test_no_llm_falls_back(self):
        events = [{"title": "公司业绩增长突破新高", "content": "超预期"}]
        result = assess_event_impact(events)
        assert "assessments" in result
        assert result.get("engine") == "keyword_rules"

    def test_empty_events_returns_error(self):
        result = assess_event_impact([])
        assert "error" in result
        assert result["assessments"] == []

    def test_with_kline_context_no_llm(self):
        events = [{"title": "公司获得重大融资", "content": "增长突破"}]
        kline_ctx = {
            "kline": [{"date": "2024-01-01", "open": 10, "close": 11, "pct_change": 5}],
            "before_avg_change_pct": 1.0,
            "after_avg_change_pct": 3.0,
            "change_delta": 2.0,
        }
        result = assess_event_impact(events, kline_context=kline_ctx)
        assert "assessments" in result


# ===================== inject_event_llm =====================


class TestInjectEventLlm:

    def test_sets_llm_ref(self):
        mock_llm = object()
        inject_event_llm(mock_llm)
        assert _llm_ref["llm"] is mock_llm
        # Cleanup
        inject_event_llm(None)

    def test_clear_llm(self):
        inject_event_llm(None)
        assert _llm_ref["llm"] is None


# ===================== Tool Definitions =====================


class TestEventImpactToolDefinitions:

    def test_tools_count(self):
        assert len(EVENT_IMPACT_TOOLS) == 3

    def test_tool_names(self):
        names = {t.name for t in EVENT_IMPACT_TOOLS}
        assert "fetch_date_events" in names
        assert "fetch_kline_context" in names
        assert "assess_event_impact" in names

    def test_fetch_date_events_tool(self):
        tool = next(t for t in EVENT_IMPACT_TOOLS if t.name == "fetch_date_events")
        assert tool.category == "data"
        assert callable(tool.callback)
        assert "date" in tool.parameters["properties"]
        assert "keyword" in tool.parameters["properties"]

    def test_fetch_kline_context_tool(self):
        tool = next(t for t in EVENT_IMPACT_TOOLS if t.name == "fetch_kline_context")
        assert tool.category == "data"
        assert "stock_code" in tool.parameters["properties"]

    def test_assess_event_impact_tool(self):
        tool = next(t for t in EVENT_IMPACT_TOOLS if t.name == "assess_event_impact")
        assert tool.category == "analysis"
        assert "events" in tool.parameters["required"]
