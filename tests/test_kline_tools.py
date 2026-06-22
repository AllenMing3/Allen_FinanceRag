"""
Test kline_tools module — STOCK_MAP, generate_kline_analysis fallback, tool definitions

Covers:
- STOCK_MAP: all entries have valid ts_code format
- generate_kline_analysis: fallback (no LLM) returns data summary
- _extract_kline_params: no-LLM passthrough
- _append_analysis_to_md: file insertion logic
- inject_kline_llm: sets reference correctly
- Tool definitions: FunctionDef shape and callback wiring
"""
import os
import tempfile
import pytest

from financial_rag.tools.kline_tools import (
    STOCK_MAP,
    generate_kline_analysis,
    _extract_kline_params,
    _append_analysis_to_md,
    inject_kline_llm,
    _kline_llm_ref,
    ANALYZE_KLINE_TOOL,
    GENERATE_KLINE_ANALYSIS_TOOL,
    KLINE_ANALYSIS_TOOLS,
    KLINE_REPORT_TOOL,
)


# ===================== STOCK_MAP =====================


class TestStockMap:

    def test_all_entries_have_valid_ts_code(self):
        """Every STOCK_MAP entry should have ts_code in format XXXXXX.XX or XXXXX.HK"""
        for key, (ts_code, name) in STOCK_MAP.items():
            assert "." in ts_code, f"{key} ts_code missing dot: {ts_code}"
            parts = ts_code.split(".")
            assert len(parts) == 2, f"{key} ts_code bad format: {ts_code}"
            assert parts[0].isdigit(), f"{key} ts_code prefix not numeric: {ts_code}"
            assert parts[1] in ("SH", "SZ", "HK"), f"{key} ts_code suffix unknown: {parts[1]}"

    def test_all_entries_have_name(self):
        for key, (ts_code, name) in STOCK_MAP.items():
            assert isinstance(name, str) and len(name) > 0, f"{key} has empty name"

    def test_known_entries(self):
        assert "茅台" in STOCK_MAP
        assert STOCK_MAP["茅台"][0] == "600519.SH"
        assert "比亚迪" in STOCK_MAP
        assert "沪深300" in STOCK_MAP

    def test_etf_codes_start_with_51_or_159(self):
        """ETF entries should have codes starting with 51 or 159"""
        etf_keys = ["沪深300", "中证500", "创业板"]
        for key in etf_keys:
            ts_code = STOCK_MAP[key][0]
            assert ts_code.startswith("51") or ts_code.startswith("159"), \
                f"{key} ETF code should start with 51 or 159: {ts_code}"


# ===================== generate_kline_analysis (fallback) =====================


class TestGenerateKlineAnalysisFallback:

    def test_no_llm_returns_fallback(self):
        # Ensure no LLM injected
        _kline_llm_ref["llm"] = None
        result = generate_kline_analysis(
            ts_code="600519.SH",
            name="贵州茅台",
            stats={"latest_close": 1800, "period_change_pct": 5.2, "up_days": 15, "down_days": 10},
            indicators={"macd": {"signal": "金叉"}, "rsi": {"value": 65, "signal": "偏强"}},
        )
        assert result["method"] == "fallback"
        assert "1800" in result["analysis"]
        assert "5.2" in result["analysis"]

    def test_no_llm_empty_stats(self):
        _kline_llm_ref["llm"] = None
        result = generate_kline_analysis(ts_code="600519.SH")
        assert result["method"] == "fallback"
        assert "N/A" in result["analysis"]

    def test_no_llm_with_kdj(self):
        _kline_llm_ref["llm"] = None
        result = generate_kline_analysis(
            ts_code="000858.SZ",
            stats={},
            indicators={"kdj": {"signal": "超买"}},
        )
        assert "KDJ" in result["analysis"]
        assert "超买" in result["analysis"]


# ===================== _extract_kline_params =====================


class TestExtractKlineParams:

    def test_no_llm_returns_query_as_keyword(self):
        keyword, days = _extract_kline_params(None, "看看茅台走势", default_days=60)
        assert keyword == "看看茅台走势"
        assert days == 60

    def test_no_llm_default_days(self):
        keyword, days = _extract_kline_params(None, "半导体ETF")
        assert keyword == "半导体ETF"
        assert days == 30  # default

    def test_no_llm_no_normalization(self):
        """Without LLM, function returns early — no topic_map normalization"""
        keyword, _ = _extract_kline_params(None, "AI")
        assert keyword == "AI"  # early return, no normalization


# ===================== _append_analysis_to_md =====================


class TestAppendAnalysisToMd:

    def test_inserts_before_basic_stats(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Test\n\n## 基础统计\n\n| 指标 | 数值 |\n")
            path = f.name
        try:
            _append_analysis_to_md(path, "## AI 分析\n趋势向好")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "## AI 技术分析" in content
            assert content.index("AI 技术分析") < content.index("基础统计")
        finally:
            os.unlink(path)

    def test_no_filepath_does_nothing(self):
        _append_analysis_to_md("", "some analysis")  # should not raise

    def test_empty_analysis_does_nothing(self):
        _append_analysis_to_md("/tmp/nonexistent.md", "")  # should not raise


# ===================== inject_kline_llm =====================


class TestInjectKlineLlm:

    def test_sets_llm_ref(self):
        mock_llm = object()
        inject_kline_llm(mock_llm)
        assert _kline_llm_ref["llm"] is mock_llm
        # Cleanup
        inject_kline_llm(None)


# ===================== Tool Definitions =====================


class TestKlineToolDefinitions:

    def test_analyze_kline_tool_shape(self):
        assert ANALYZE_KLINE_TOOL.name == "analyze_kline"
        assert ANALYZE_KLINE_TOOL.category == "analysis"
        assert callable(ANALYZE_KLINE_TOOL.callback)
        assert "ts_code" in ANALYZE_KLINE_TOOL.parameters["properties"]

    def test_generate_kline_analysis_tool_shape(self):
        assert GENERATE_KLINE_ANALYSIS_TOOL.name == "generate_kline_analysis"
        assert callable(GENERATE_KLINE_ANALYSIS_TOOL.callback)
        required = GENERATE_KLINE_ANALYSIS_TOOL.parameters["required"]
        assert "ts_code" in required
        assert "stats" in required

    def test_kline_report_tool_shape(self):
        assert KLINE_REPORT_TOOL.name == "fetch_kline_report"
        assert KLINE_REPORT_TOOL.category == "data"
        assert "keyword" in KLINE_REPORT_TOOL.parameters["required"]

    def test_kline_analysis_tools_list(self):
        assert len(KLINE_ANALYSIS_TOOLS) == 2
        names = {t.name for t in KLINE_ANALYSIS_TOOLS}
        assert "analyze_kline" in names
        assert "generate_kline_analysis" in names
