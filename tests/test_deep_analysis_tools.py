"""
Test analysis_tools module — analyze_news_deep, analyze_topic_deep wrappers

Covers:
- analyze_news_deep: empty text returns error
- analyze_news_deep: valid text (no LLM) returns structured result
- analyze_topic_deep: empty topic returns error
- inject_analysis_deps: sets refs correctly
- Tool definitions: ANALYSIS_TOOLS shape
"""
import pytest

from financial_rag.tools.analysis_tools import (
    analyze_news_deep,
    analyze_topic_deep,
    inject_analysis_deps,
    _analysis_refs,
    ANALYSIS_TOOLS,
)


# ===================== inject_analysis_deps =====================


class TestInjectAnalysisDeps:

    def test_sets_all_refs(self):
        mock_llm = object()
        mock_retriever = object()
        inject_analysis_deps(llm=mock_llm, retriever=mock_retriever, kb_built=True)
        assert _analysis_refs["llm"] is mock_llm
        assert _analysis_refs["retriever"] is mock_retriever
        assert _analysis_refs["kb_built"] is True
        # Cleanup
        inject_analysis_deps(llm=None, retriever=None, kb_built=False)

    def test_defaults(self):
        inject_analysis_deps()
        assert _analysis_refs["llm"] is None
        assert _analysis_refs["retriever"] is None
        assert _analysis_refs["kb_built"] is False


# ===================== analyze_news_deep =====================


class TestAnalyzeNewsDeep:

    def setup_method(self):
        inject_analysis_deps(llm=None, retriever=None, kb_built=False)

    def test_empty_text_returns_error(self):
        result = analyze_news_deep(text="")
        assert "error" in result
        assert result["structured"] == {}

    def test_none_text_returns_error(self):
        result = analyze_news_deep(text=None)
        assert "error" in result

    def test_valid_text_no_llm(self):
        text = "英伟达正式发布Blackwell B200 GPU，单卡AI训练性能较上一代提升4倍。微软Azure已部署10万张B200用于训练GPT-5。"
        result = analyze_news_deep(text=text)
        # Should return some structured analysis from heuristic path
        assert isinstance(result, dict)
        assert "error" not in result or result.get("analysis")

    def test_with_query_parameter(self):
        text = "央行宣布降准50个基点，释放流动性约1万亿元。"
        result = analyze_news_deep(text=text, query="央行降准影响")
        assert isinstance(result, dict)


# ===================== analyze_topic_deep =====================


class TestAnalyzeTopicDeep:

    def setup_method(self):
        inject_analysis_deps(llm=None, retriever=None, kb_built=False)

    def test_empty_topic_returns_error(self):
        result = analyze_topic_deep(topic="")
        assert "error" in result
        assert result["structured"] == {}

    def test_none_topic_returns_error(self):
        result = analyze_topic_deep(topic=None)
        assert "error" in result


# ===================== Tool Definitions =====================


class TestAnalysisToolDefinitions:

    def test_tools_count(self):
        assert len(ANALYSIS_TOOLS) == 2

    def test_tool_names(self):
        names = {t.name for t in ANALYSIS_TOOLS}
        assert "analyze_news_deep" in names
        assert "analyze_topic_deep" in names

    def test_analyze_news_deep_tool(self):
        tool = next(t for t in ANALYSIS_TOOLS if t.name == "analyze_news_deep")
        assert tool.category == "analysis"
        assert callable(tool.callback)
        assert "text" in tool.parameters["required"]

    def test_analyze_topic_deep_tool(self):
        tool = next(t for t in ANALYSIS_TOOLS if t.name == "analyze_topic_deep")
        assert tool.category == "analysis"
        assert "topic" in tool.parameters["required"]
        assert "max_news" in tool.parameters["properties"]
