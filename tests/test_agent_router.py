"""
Test AgentRouter — intent classification, chain selection, metadata extraction

Covers:
- 5 intent types: kline, event_impact, report, news, general
- Metadata extraction: date, stock_code, stock_name
- Runtime intent registration
- Confidence scoring
- Edge cases: ambiguous queries, empty queries, multi-intent queries
"""
import pytest
from financial_rag.core.agent_router import AgentRouter, create_agent_router, QueryIntent, RoutingDecision


class TestIntentClassification:
    """Intent classification — keyword + regex matching"""

    @pytest.fixture
    def router(self):
        return create_agent_router()

    # ---- kline intent ----

    def test_kline_keyword(self, router):
        d = router.route("茅台最近走势怎么样")
        assert d.intent == "kline"
        assert d.confidence > 0

    def test_kline_technical_term(self, router):
        d = router.route("帮我分析一下MACD指标")
        assert d.intent == "kline"

    def test_kline_stock_code_pattern(self, router):
        d = router.route("600519的K线走势")
        assert d.intent == "kline"

    def test_kline_stock_name_pattern(self, router):
        d = router.route("比亚迪的技术分析")
        assert d.intent == "kline"

    def test_kline_support_resistance(self, router):
        d = router.route("茅台支撑位和压力位在哪里")
        assert d.intent == "kline"

    # ---- event_impact intent ----

    def test_event_impact_date(self, router):
        d = router.route("2024年6月1日发生了什么大事")
        assert d.intent == "event_impact"
        assert d.confidence >= 0.8  # pattern + keyword match

    def test_event_impact_keyword(self, router):
        d = router.route("这个消息是利好还是利空")
        assert d.intent == "event_impact"

    def test_event_impact_merger(self, router):
        d = router.route("某公司并购重组影响分析")
        assert d.intent == "event_impact"

    def test_event_impact_limit(self, router):
        d = router.route("今天涨停的股票有哪些")
        assert d.intent == "event_impact"

    # ---- report intent ----

    def test_report_financial(self, router):
        d = router.route("商汤科技2024年营收多少")
        assert d.intent == "report"

    def test_report_annual(self, router):
        d = router.route("贵州茅台年报分析")
        assert d.intent == "report"

    def test_report_metrics(self, router):
        d = router.route("比亚迪的净利润增长率")
        assert d.intent == "report"

    # ---- news intent ----

    def test_news_latest(self, router):
        d = router.route("今天有什么AI新闻")
        assert d.intent == "news"

    def test_news_industry(self, router):
        d = router.route("最新的行业动态资讯")
        assert d.intent == "news"

    # ---- general intent (fallback) ----

    def test_general_query(self, router):
        d = router.route("今天天气怎么样")
        assert d.intent == "general"
        assert d.confidence == 0.5  # default confidence for general

    def test_general_empty(self, router):
        d = router.route("")
        assert d.intent == "general"

    def test_general_vague(self, router):
        d = router.route("帮我分析一下")
        assert d.intent == "general"


class TestMetadataExtraction:
    """Metadata extraction — date, stock_code, stock_name"""

    @pytest.fixture
    def router(self):
        return create_agent_router()

    def test_extract_date_dash(self, router):
        d = router.route("2024-06-01 发生了什么")
        assert d.metadata.get("date") == "2024-06-01"

    def test_extract_date_chinese(self, router):
        d = router.route("2024年6月1日有什么大事")
        assert d.metadata.get("date") == "2024-06-01"

    def test_extract_date_compact(self, router):
        d = router.route("20240601有什么新闻")
        assert d.metadata.get("date") == "2024-06-01"

    def test_extract_stock_keyword(self, router):
        d = router.route("茅台走势分析")
        assert d.metadata.get("ts_code") == "600519.SH"
        assert d.metadata.get("stock_name") == "贵州茅台"

    def test_extract_stock_code_sh(self, router):
        d = router.route("600519最近怎么样")
        assert d.metadata.get("stock_code") == "600519.SH"

    def test_extract_stock_code_sz(self, router):
        d = router.route("000858走势如何")
        assert d.metadata.get("stock_code") == "000858.SZ"

    def test_no_metadata(self, router):
        d = router.route("今天天气怎么样")
        assert "date" not in d.metadata or d.metadata["date"] == ""

    def test_metadata_context_override(self, router):
        """External context takes precedence"""
        d = router.route("分析茅台走势", context={"ts_code": "999999.SH"})
        assert d.metadata.get("ts_code") == "999999.SH"


class TestChainSelection:
    """Agent chain selection for each intent"""

    @pytest.fixture
    def router(self):
        return create_agent_router()

    def test_kline_chain(self, router):
        d = router.route("茅台K线走势")
        assert "AnalysisAgent" in d.agent_chain
        assert "ScoringAgent" in d.agent_chain

    def test_event_impact_chain(self, router):
        d = router.route("2024年6月1日发生了什么大事")
        assert "AnalysisAgent" in d.agent_chain
        assert "ScoringAgent" in d.agent_chain

    def test_report_chain(self, router):
        d = router.route("商汤科技2024年营收多少")
        assert "IngestionAgent" in d.agent_chain
        assert "AnalysisAgent" in d.agent_chain

    def test_general_chain(self, router):
        d = router.route("随便聊聊")
        assert "IngestionAgent" in d.agent_chain
        assert "AnalysisAgent" in d.agent_chain
        assert "ScoringAgent" in d.agent_chain

    def test_no_coordinator_in_chain(self, router):
        """CoordinatorAgent should NOT be in any chain (pipeline routes)"""
        intent_map = router.get_intent_map()
        for intent, chain in intent_map.items():
            assert "CoordinatorAgent" not in chain, f"CoordinatorAgent found in {intent} chain"


class TestRuntimeRegistration:
    """Runtime intent registration"""

    @pytest.fixture
    def router(self):
        return create_agent_router()

    def test_register_new_intent(self, router):
        router.register_intent(
            name="options",
            keywords=["期权", "看涨期权", "看跌期权"],
            chain=["KLineAgent", "ReportAgent", "ScoringAgent"],
        )
        d = router.route("这个看涨期权怎么分析")
        assert d.intent == "options"

    def test_set_default_chain(self, router):
        router.set_default_chain(["ReportAgent"])
        d = router.route("随便聊点什么")
        assert d.agent_chain == ["ReportAgent"]

    def test_get_intent_map(self, router):
        m = router.get_intent_map()
        assert "kline" in m
        assert "event_impact" in m
        assert "report" in m
        assert "news" in m
        assert "_default" in m


class TestRoutingDecision:
    """RoutingDecision data class"""

    def test_str_representation(self):
        d = RoutingDecision(
            intent="kline",
            agent_chain=["KLineAgent", "ReportAgent"],
            confidence=0.8,
        )
        s = str(d)
        assert "kline" in s
        assert "80%" in s
        assert "KLineAgent" in s

    def test_metadata_default_empty(self):
        d = RoutingDecision(intent="general", agent_chain=[], confidence=0.5)
        assert d.metadata == {}

    def test_classify_only(self):
        router = create_agent_router()
        intent = router.classify("茅台走势")
        assert isinstance(intent, QueryIntent)
        assert intent.name == "kline"
