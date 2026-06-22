"""
Test report_tools module — synthesize_report fallback, _format_sources, _fallback_report

Covers:
- synthesize_report: fallback with sources, no data error
- _format_sources: text truncation, non-dict filtering
- _fallback_report: structure, entities normalization
- inject_report_llm: sets reference
- Tool definitions: REPORT_TOOLS shape
"""
import pytest

from financial_rag.tools.report_tools import (
    synthesize_report,
    _format_sources,
    _fallback_report,
    inject_report_llm,
    _report_llm_ref,
    SYNTHESIZE_REPORT_TOOL,
    REPORT_TOOLS,
)


# ===================== _format_sources =====================


class TestFormatSources:

    def test_basic_formatting(self):
        sources = [
            {"id": 1, "title": "商汤年报", "text": "营收50.3亿元", "source": "test", "date": "2024-01-01"},
        ]
        result = _format_sources(sources)
        assert "商汤年报" in result
        assert "2024-01-01" in result
        assert "营收50.3亿元" in result

    def test_text_truncation_at_800(self):
        sources = [
            {"id": 1, "title": "Long doc", "text": "A" * 1000, "source": "test"},
        ]
        result = _format_sources(sources)
        # Text should be truncated at 800 chars
        assert len(result) < 1200  # 800 + header overhead

    def test_total_truncation_at_12000(self):
        sources = [
            {"id": i, "title": f"Doc {i}", "text": "B" * 790, "source": "test"}
            for i in range(20)
        ]
        result = _format_sources(sources)
        assert "截断" in result or len(result) <= 12100

    def test_non_dict_items_filtered(self):
        sources = [{"id": 1, "title": "Valid", "text": "ok"}, "not_a_dict", 42]
        result = _format_sources(sources)
        assert "Valid" in result
        # Non-dict items should be skipped silently

    def test_empty_sources(self):
        result = _format_sources([])
        assert result == ""

    def test_missing_fields(self):
        sources = [{"text": "just text"}]
        result = _format_sources(sources)
        assert "just text" in result


# ===================== _fallback_report =====================


class TestFallbackReport:

    def test_structure_with_sources(self):
        sources = [
            {"id": 1, "text": "商汤营收50亿", "title": "商汤年报", "source": "test"},
            {"id": 2, "text": "AI增长30%", "title": "行业报告", "source": "test"},
        ]
        result = _fallback_report("AI行业分析", sources, {"revenue": 50}, [])
        assert result["method"] == "fallback"
        report = result["report"]
        assert report["title"] == "AI行业分析"
        assert len(report["key_findings"]) == 2
        assert "2" in report["trend_analysis"]  # "2 条相关新闻"
        assert report["sentiment"]["overall"] == "neutral"

    def test_with_company_entities(self):
        entities = [
            {"type": "company", "data": {"name": "商汤科技"}},
            {"type": "company", "data": {"name": "英伟达"}},
        ]
        result = _fallback_report("test", [{"id": 1, "text": "x", "title": "x"}], {}, entities)
        assert "商汤科技" in result["report"]["affected_companies"]
        assert "英伟达" in result["report"]["affected_companies"]

    def test_empty_sources(self):
        result = _fallback_report("test", [], {}, [])
        assert result["report"]["key_findings"] == []
        assert "0" in result["report"]["trend_analysis"]

    def test_query_default_title(self):
        result = _fallback_report("", [{"id": 1, "text": "x", "title": "x"}], {}, [])
        assert result["report"]["title"] == "新闻分析报告"

    def test_max_5_findings(self):
        sources = [{"id": i, "text": f"text_{i}", "title": f"title_{i}"} for i in range(10)]
        result = _fallback_report("test", sources, {}, [])
        assert len(result["report"]["key_findings"]) == 5

    def test_max_5_companies(self):
        entities = [{"type": "company", "data": {"name": f"Company_{i}"}} for i in range(10)]
        result = _fallback_report("test", [{"id": 1, "text": "x", "title": "x"}], {}, entities)
        assert len(result["report"]["affected_companies"]) == 5

    def test_hallucination_check_empty(self):
        result = _fallback_report("test", [], {}, [])
        assert result["hallucination_check"] == {}


# ===================== synthesize_report =====================


class TestSynthesizeReport:

    def setup_method(self):
        _report_llm_ref["llm"] = None

    def test_no_data_returns_error(self):
        result = synthesize_report(query="test")
        assert "error" in result
        assert result["method"] == "none"

    def test_fallback_with_sources(self):
        result = synthesize_report(
            query="AI行业",
            sources=[{"id": 1, "text": "AI增长", "title": "新闻1", "source": "test"}],
            metrics={"growth": 30},
        )
        assert result["method"] == "fallback"
        assert "report" in result

    def test_entities_dict_to_list_normalization(self):
        """When entities is a dict (from extract_entities), it should be normalized to list.
        Note: normalized items have data as list, so _fallback_report may not extract company names."""
        entities_dict = {
            "company": [{"name": "商汤科技"}],
            "product": [{"name": "日日新"}],
        }
        result = synthesize_report(
            query="test",
            sources=[{"id": 1, "text": "text", "title": "title"}],
            entities=entities_dict,
        )
        assert result["method"] == "fallback"
        # After normalization, entities are [{"type": "company", "data": [...]}, ...]
        # With the fix, _fallback_report extracts names from list data too
        assert "商汤科技" in result["report"]["affected_companies"]

    def test_fallback_with_metrics_only(self):
        result = synthesize_report(
            query="test",
            metrics={"revenue": 100, "profit": 20},
        )
        assert result["method"] == "fallback"


# ===================== inject_report_llm =====================


class TestInjectReportLlm:

    def test_sets_llm_ref(self):
        mock_llm = object()
        inject_report_llm(mock_llm)
        assert _report_llm_ref["llm"] is mock_llm
        # Cleanup
        inject_report_llm(None)

    def test_clear_llm(self):
        inject_report_llm(None)
        assert _report_llm_ref["llm"] is None


# ===================== Tool Definitions =====================


class TestReportToolDefinitions:

    def test_tools_count(self):
        assert len(REPORT_TOOLS) == 1

    def test_synthesize_report_tool(self):
        assert SYNTHESIZE_REPORT_TOOL.name == "synthesize_report"
        assert SYNTHESIZE_REPORT_TOOL.category == "analysis"
        assert callable(SYNTHESIZE_REPORT_TOOL.callback)

    def test_tool_parameters(self):
        props = SYNTHESIZE_REPORT_TOOL.parameters["properties"]
        assert "query" in props
        assert "sources" in props
        assert "metrics" in props
        assert "entities" in props

    def test_tool_has_tags(self):
        assert "报告" in SYNTHESIZE_REPORT_TOOL.tags
        assert "LLM" in SYNTHESIZE_REPORT_TOOL.tags
