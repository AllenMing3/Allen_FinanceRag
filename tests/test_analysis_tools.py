"""
Test analysis tools — pure computation (no LLM, no data sources)

Tests calculate_growth_rate, calculate_financial_ratio, compare_metrics, summarize_financials.
"""
import pytest
from financial_rag.tools.core import (
    calculate_growth_rate,
    calculate_financial_ratio,
    compare_metrics,
    summarize_financials,
    FunctionRegistry,
    ToolExecutor,
    ToolCallRequest,
    create_financial_registry,
)


class TestCalculateGrowthRate:
    def test_positive_growth(self):
        result = calculate_growth_rate(50.3, 36.9, label="revenue")
        assert result["direction"] == "增长"
        assert result["growth_rate"] > 0
        assert result["label"] == "revenue"

    def test_negative_growth(self):
        result = calculate_growth_rate(30.0, 50.0, label="profit")
        assert result["direction"] == "下降"
        assert result["growth_rate"] < 0

    def test_zero_growth(self):
        result = calculate_growth_rate(100.0, 100.0)
        assert result["direction"] == "持平"
        assert result["growth_rate"] == 0.0

    def test_previous_zero(self):
        result = calculate_growth_rate(50.0, 0)
        assert result["growth_rate"] is None
        assert "error" in result

    def test_absolute_change(self):
        result = calculate_growth_rate(50.3, 36.9)
        assert abs(result["absolute_change"] - 13.4) < 0.1


class TestCalculateFinancialRatio:
    def test_gross_margin(self):
        result = calculate_financial_ratio("毛利率", 30.0, 100.0)
        assert result["value"] == "30.0%"

    def test_zero_denominator(self):
        result = calculate_financial_ratio("ROE", 10.0, 0)
        assert result["value"] is None
        assert "error" in result

    def test_custom_unit(self):
        result = calculate_financial_ratio("效率", 85, 100, unit="分")
        assert "分" in result["value"]


class TestCompareMetrics:
    def test_basic_comparison(self):
        a = {"name": "商汤科技", "revenue": 50.3, "rd_expense": 18.7}
        b = {"name": "智谱AI", "revenue": 15.0, "rd_expense": 10.0}
        result = compare_metrics(a, b, ["revenue", "rd_expense"])

        assert result["company_a"] == "商汤科技"
        assert result["company_b"] == "智谱AI"
        assert len(result["comparisons"]) == 2
        assert result["comparisons"][0]["diff_percent"] > 0

    def test_missing_metric(self):
        a = {"name": "A", "revenue": 100}
        b = {"name": "B"}
        result = compare_metrics(a, b, ["revenue"])
        assert "error" in result["comparisons"][0]


class TestSummarizeFinancials:
    def test_basic_summary(self):
        metrics = {"营收_亿元": 50.3, "净利润_亿元": -42.1}
        result = summarize_financials(metrics, company_name="商汤科技", period="2024年")
        assert "商汤科技" in result
        assert "50.3" in result

    def test_empty_metrics(self):
        result = summarize_financials({})
        assert isinstance(result, str)


class TestRegistryAndExecutor:
    """Test the tool registry + executor infrastructure"""

    def test_registry_has_all_tools(self):
        registry = create_financial_registry()
        assert len(registry) >= 15  # 15 registered tools

        # Check key tools exist
        assert registry.can_handle("extract_financial_metrics")
        assert registry.can_handle("extract_entities")
        assert registry.can_handle("extract_document_metadata")
        assert registry.can_handle("detect_document_type")
        assert registry.can_handle("generate_search_queries")
        assert registry.can_handle("calculate_growth_rate")
        assert registry.can_handle("calculate_financial_ratio")
        assert registry.can_handle("compare_metrics")
        assert registry.can_handle("summarize_financials")
        assert registry.can_handle("search_financial_data")
        assert registry.can_handle("fetch_stock_news")
        assert registry.can_handle("fetch_financial_news")

    def test_executor_runs_tool(self):
        registry = create_financial_registry()
        executor = ToolExecutor(registry)

        request = ToolCallRequest(
            id="test_1",
            name="calculate_growth_rate",
            arguments={"current_value": 50.3, "previous_value": 36.9, "label": "revenue"},
        )
        result = executor.execute(request)
        assert result.success
        assert result.result["growth_rate"] > 0

    def test_executor_unknown_tool(self):
        registry = create_financial_registry()
        executor = ToolExecutor(registry)

        request = ToolCallRequest(id="test_2", name="nonexistent_tool", arguments={})
        result = executor.execute(request)
        assert not result.success
        assert "未知能力" in result.error

    def test_openai_schemas(self):
        registry = create_financial_registry()
        schemas = registry.to_openai_schemas()
        assert len(schemas) >= 15
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]

    def test_list_by_category(self):
        registry = create_financial_registry()
        extraction_tools = registry.list_by_category("extraction")
        assert len(extraction_tools) >= 5
        analysis_tools = registry.list_by_category("analysis")
        assert len(analysis_tools) >= 4
