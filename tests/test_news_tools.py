"""
Test news_tools module — _extract_keywords, EN_TO_CN, _GENERIC_KEYWORDS, tool definitions

Covers:
- _extract_keywords: no-LLM returns [query]
- EN_TO_CN: mapping completeness
- _GENERIC_KEYWORDS: filtering broad terms
- _append_summary_to_md: file insertion logic
- _generate_summary: no-LLM returns empty
- Tool definitions: NEWS_REPORT_TOOL shape
"""
import os
import tempfile
import pytest

from financial_rag.tools.news_tools import (
    _extract_keywords,
    _generate_summary,
    _append_summary_to_md,
    EN_TO_CN,
    TOPIC_MAP,
    _GENERIC_KEYWORDS,
    NEWS_REPORT_TOOL,
)


# ===================== _extract_keywords (no LLM) =====================


class TestExtractKeywordsNoLLM:

    def test_returns_query_as_first_keyword(self):
        result = _extract_keywords(None, "英伟达 Blackwell GPU")
        assert result == ["英伟达 Blackwell GPU"]

    def test_empty_query(self):
        result = _extract_keywords(None, "")
        assert result == [""]

    def test_chinese_query(self):
        result = _extract_keywords(None, "央行降准")
        assert result[0] == "央行降准"


# ===================== EN_TO_CN Mapping =====================


class TestEnToCnMapping:

    def test_ai_mapping(self):
        assert EN_TO_CN["ai"] == "人工智能"
        assert EN_TO_CN["AI"] == "人工智能"

    def test_known_entries(self):
        assert "robot" in EN_TO_CN
        assert "blockchain" in EN_TO_CN
        assert "ev" in EN_TO_CN

    def test_all_values_are_chinese(self):
        for en, cn in EN_TO_CN.items():
            # Chinese chars are typically > \u4e00
            assert any("\u4e00" <= c <= "\u9fff" for c in cn), \
                f"EN_TO_CN['{en}'] = '{cn}' doesn't contain Chinese chars"


# ===================== TOPIC_MAP =====================


class TestTopicMap:

    def test_ai_entries(self):
        assert TOPIC_MAP["AI"] == "人工智能"
        assert TOPIC_MAP["ai"] == "人工智能"

    def test_tech_entries(self):
        assert "芯片" in TOPIC_MAP
        assert "半导体" in TOPIC_MAP
        assert "5G" in TOPIC_MAP

    def test_all_values_non_empty(self):
        for key, val in TOPIC_MAP.items():
            assert isinstance(val, str) and len(val) > 0


# ===================== _GENERIC_KEYWORDS =====================


class TestGenericKeywords:

    def test_filters_broad_terms(self):
        assert "AI" in _GENERIC_KEYWORDS
        assert "科技" in _GENERIC_KEYWORDS
        assert "最新" in _GENERIC_KEYWORDS
        assert "新闻" in _GENERIC_KEYWORDS

    def test_specific_terms_not_filtered(self):
        assert "茅台" not in _GENERIC_KEYWORDS
        assert "英伟达" not in _GENERIC_KEYWORDS
        assert "降准" not in _GENERIC_KEYWORDS

    def test_set_type(self):
        assert isinstance(_GENERIC_KEYWORDS, set)


# ===================== _generate_summary =====================


class TestGenerateSummary:

    def test_no_llm_returns_empty(self):
        result = _generate_summary(None, [{"title": "test"}], "topic")
        assert result == ""

    def test_empty_headlines_returns_empty(self):
        result = _generate_summary(None, [], "topic")
        assert result == ""


# ===================== _append_summary_to_md =====================


class TestAppendSummaryToMd:

    def test_inserts_after_first_separator(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# News\n> Date\n\n---\n\n## 1. Title\n> Content\n")
            path = f.name
        try:
            _append_summary_to_md(path, "AI行业最新动态摘要")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "AI行业最新动态摘要" in content
            assert "AI 摘要" in content
        finally:
            os.unlink(path)

    def test_no_filepath_does_nothing(self):
        _append_summary_to_md("", "some summary")  # should not raise

    def test_empty_summary_does_nothing(self):
        _append_summary_to_md("/tmp/nonexistent.md", "")  # should not raise

    def test_no_separator_no_change(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Just a title\nNo separators here\n")
            path = f.name
        try:
            _append_summary_to_md(path, "Summary text")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Only 1 part (no split), so no insertion
            assert "AI 摘要" not in content
        finally:
            os.unlink(path)


# ===================== Tool Definitions =====================


class TestNewsToolDefinitions:

    def test_news_report_tool_name(self):
        assert NEWS_REPORT_TOOL.name == "fetch_news_report"

    def test_news_report_tool_category(self):
        assert NEWS_REPORT_TOOL.category == "data"

    def test_news_report_tool_required(self):
        assert "keyword" in NEWS_REPORT_TOOL.parameters["required"]

    def test_news_report_tool_callback(self):
        assert callable(NEWS_REPORT_TOOL.callback)

    def test_news_report_tool_has_tags(self):
        assert "新闻" in NEWS_REPORT_TOOL.tags
