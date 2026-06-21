"""
Test analysis service — analyze_news_text + analyze_topic_research with mock mode
"""
import pytest
from unittest.mock import patch

from tests.conftest import (
    SAMPLE_AI_FINANCIAL_REPORT,
    SAMPLE_AI_NEWS,
    SAMPLE_AI_FUNDING,
    SAMPLE_AI_PRODUCT_LAUNCH,
)


class TestAnalyzeNewsText:
    """analyze_news_text: paste news → extraction + verdict (no external API needed)"""

    def test_basic_structure(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_NEWS)
        assert "assessment" in result
        assert "analysis" in result
        assert "structured" in result
        assert "metrics" in result
        assert "entities" in result
        assert "doc_type" in result
        assert "kb_sources" in result

    def test_structured_output_shape(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_NEWS)
        s = result["structured"]
        assert isinstance(s, dict)
        assert "verdict" in s
        assert "impact" in s
        assert "key_signals" in s
        assert "analysis" in s
        assert "risks" in s
        assert "watch_next" in s
        # Impact should have 4 dimensions
        imp = s["impact"]
        for dim in ("industry", "company", "tech", "market"):
            assert dim in imp
            assert "direction" in imp[dim]
            assert "summary" in imp[dim]
        # Signals should be a list
        assert isinstance(s["key_signals"], list)
        assert len(s["key_signals"]) > 0
        for sig in s["key_signals"]:
            assert "signal" in sig
            assert "severity" in sig
            assert "type" in sig

    def test_assessment_value(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_NEWS)
        assert result["assessment"] in ("bullish", "bearish", "neutral")

    def test_extraction_metrics(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_FINANCIAL_REPORT)
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        # Should extract at least some financial numbers via regex
        assert len(metrics) > 0 or len(result["entities"]) > 0

    def test_extraction_entities(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_FINANCIAL_REPORT)
        entities = result["entities"]
        assert isinstance(entities, dict)

    def test_doc_type_detection(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_FINANCIAL_REPORT)
        assert result["doc_type"] is not None

    def test_funding_text(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_FUNDING)
        assert result["assessment"] in ("bullish", "bearish", "neutral")
        assert result["analysis"]

    def test_product_launch_text(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_PRODUCT_LAUNCH)
        assert result["assessment"] in ("bullish", "bearish", "neutral")

    def test_empty_kb(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_NEWS, kb_built=False, retriever=None)
        assert result["kb_sources"] == []

    def test_with_query(self):
        from financial_rag.services.analysis import analyze_news_text
        result = analyze_news_text(SAMPLE_AI_NEWS, query="英伟达 GPU")
        assert result["assessment"] in ("bullish", "bearish", "neutral")


class TestAnalyzeTopicResearch:
    """analyze_topic_research: topic → mock news → verdict"""

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_mock_mode_basic(self, _mock_flag):
        from financial_rag.services.analysis import analyze_topic_research
        result = analyze_topic_research("AI大模型")
        assert result["assessment"] in ("bullish", "bearish", "neutral")
        assert result["topic"] == "AI大模型"
        assert result["news_count"] > 0
        assert len(result["news"]) > 0

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_mock_mode_structure(self, _mock_flag):
        from financial_rag.services.analysis import analyze_topic_research
        result = analyze_topic_research("半导体")
        # Verify all expected keys
        assert "assessment" in result
        assert "analysis" in result
        assert "structured" in result
        assert "topic" in result
        assert "news_count" in result
        assert "news" in result
        assert "kb_sources" in result

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_structured_output_shape(self, _mock_flag):
        from financial_rag.services.analysis import analyze_topic_research
        result = analyze_topic_research("AI大模型")
        s = result["structured"]
        assert isinstance(s, dict)
        assert "verdict" in s
        assert "sub_topics" in s
        assert "key_players" in s
        assert "sentiment_trend" in s
        assert "analysis" in s
        assert "risks" in s
        assert isinstance(s["sub_topics"], list)
        assert isinstance(s["key_players"], list)
        assert s["sentiment_trend"] in ("improving", "deteriorating", "stable", "mixed")

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_mock_mode_news_items(self, _mock_flag):
        from financial_rag.services.analysis import analyze_topic_research
        result = analyze_topic_research("芯片")
        for item in result["news"]:
            assert "title" in item
            assert "source" in item
            assert "publish_time" in item

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_mock_mode_max_news(self, _mock_flag):
        from financial_rag.services.analysis import analyze_topic_research
        result = analyze_topic_research("AI", max_news=5)
        assert result["news_count"] <= 30  # mock pool size, not exact limit
        assert len(result["news"]) <= 10  # capped at 10 for display

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_mock_mode_no_llm(self, _mock_flag):
        """Without LLM, should still return structured result"""
        from financial_rag.services.analysis import analyze_topic_research
        result = analyze_topic_research("GPU", llm=None, retriever=None, kb_built=False)
        assert result["assessment"] in ("bullish", "bearish", "neutral")
        assert "话题" in result["analysis"] or "新闻" in result["analysis"] or result["news_count"] > 0

    @patch("financial_rag.config.is_mock_enabled", return_value=True)
    def test_mock_mode_different_topics(self, _mock_flag):
        """Different topics should all work with mock data"""
        from financial_rag.services.analysis import analyze_topic_research
        for topic in ["DeepSeek", "英伟达", "大模型", "算力"]:
            result = analyze_topic_research(topic)
            assert result["topic"] == topic
            assert result["assessment"] in ("bullish", "bearish", "neutral")


class TestAnalysisHelpers:
    """Internal helper functions"""

    def test_parse_verdict_bullish(self):
        from financial_rag.services.analysis import _parse_verdict
        assert _parse_verdict("【判断】利好") == "bullish"

    def test_parse_verdict_bearish(self):
        from financial_rag.services.analysis import _parse_verdict
        assert _parse_verdict("【判断】利空") == "bearish"

    def test_parse_verdict_neutral(self):
        from financial_rag.services.analysis import _parse_verdict
        assert _parse_verdict("【判断】中性") == "neutral"

    def test_heuristic_assessment_positive(self):
        from financial_rag.services.analysis import _heuristic_assessment
        metrics = {"revenue": {"value": "50亿", "yoy_growth": 36}}
        assessment, structured = _heuristic_assessment("financial_report", metrics, {})
        assert assessment in ("bullish", "neutral")
        assert isinstance(structured, dict)
        assert "impact" in structured
        assert "key_signals" in structured

    def test_heuristic_assessment_negative(self):
        from financial_rag.services.analysis import _heuristic_assessment
        metrics = {"revenue": {"value": "10亿", "yoy_growth": -30}}
        assessment, structured = _heuristic_assessment("financial_report", metrics, {})
        assert assessment in ("bearish", "neutral")
        assert isinstance(structured, dict)

    def test_search_kb_no_retriever(self):
        from financial_rag.services.analysis import _search_kb
        assert _search_kb(None, "test", True) == []

    def test_search_kb_not_built(self):
        from financial_rag.services.analysis import _search_kb
        assert _search_kb(object(), "test", False) == []
