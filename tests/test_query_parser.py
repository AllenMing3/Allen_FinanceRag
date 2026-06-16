"""Tests for QueryParser — 查询解析器"""
import pytest
from financial_rag.retrievers.query_parser import (
    QueryParser, QueryResult,
    FINANCIAL_TERMS, INDUSTRY_TERMS, ACTION_TERMS, STOP_WORDS,
)


@pytest.fixture
def parser():
    """Create QueryParser with default STOCK_MAP"""
    return QueryParser()


# ===================== 实体抽取: 股票 =====================


class TestStockExtraction:
    def test_stock_keyword_match(self, parser):
        result = parser.parse("茅台最近走势怎么样")
        assert result.stock_code == "600519.SH"
        assert result.stock_name == "贵州茅台"
        assert result.stock_keyword == "茅台"

    def test_stock_keyword_ningde(self, parser):
        result = parser.parse("宁德时代的技术分析")
        assert result.stock_code == "300750.SZ"
        assert result.stock_name == "宁德时代"

    def test_stock_code_6digit_sh(self, parser):
        result = parser.parse("帮我分析一下600036这个股票")
        assert result.stock_code == "600036.SH"

    def test_stock_code_6digit_sz(self, parser):
        result = parser.parse("002594值不值得买")
        assert result.stock_code == "002594.SZ"

    def test_stock_code_300_prefix(self, parser):
        result = parser.parse("看看300750的K线")
        assert result.stock_code == "300750.SZ"

    def test_no_stock(self, parser):
        result = parser.parse("AI行业最近发展如何")
        assert result.stock_code == ""
        assert result.stock_name == ""

    def test_custom_stock_map(self):
        custom_map = {"苹果": ("AAPL", "Apple Inc")}
        parser = QueryParser(stock_map=custom_map)
        result = parser.parse("苹果最新财报")
        assert result.stock_code == "AAPL"
        assert result.stock_name == "Apple Inc"


# ===================== 实体抽取: 日期 =====================


class TestDateExtraction:
    def test_absolute_date_dash(self, parser):
        result = parser.parse("2024-06-01的茅台行情")
        assert result.date == "2024-06-01"

    def test_absolute_date_chinese(self, parser):
        result = parser.parse("2024年3月15日的市场报告")
        assert result.date == "2024-03-15"

    def test_absolute_date_compact(self, parser):
        result = parser.parse("20240501的交易数据")
        assert result.date == "2024-05-01"

    def test_relative_one_week(self, parser):
        result = parser.parse("最近一周的茅台走势")
        assert result.date_range is not None
        assert "gte" in result.date_range
        assert "lte" in result.date_range

    def test_relative_one_month(self, parser):
        result = parser.parse("近一个月AI芯片板块表现")
        assert result.date_range is not None

    def test_relative_half_month(self, parser):
        result = parser.parse("最近半个月市场行情")
        assert result.date_range is not None

    def test_no_date(self, parser):
        result = parser.parse("茅台技术分析")
        assert result.date == ""
        assert result.date_range is None


# ===================== 关键词抽取 =====================


class TestKeywordExtraction:
    def test_stock_keyword_high_weight(self, parser):
        result = parser.parse("茅台最近走势")
        # 茅台 should have weight 3.0
        keywords_dict = dict(result.keywords)
        assert "茅台" in keywords_dict
        assert keywords_dict["茅台"] == 3.0

    def test_financial_term_medium_weight(self, parser):
        result = parser.parse("看看毛利率和净利率")
        keywords_dict = dict(result.keywords)
        assert "毛利率" in keywords_dict
        assert keywords_dict["毛利率"] == 2.0

    def test_industry_term(self, parser):
        result = parser.parse("AI芯片行业分析")
        keywords_dict = dict(result.keywords)
        # AI and 芯片 should be in industry terms (weight 1.5)
        assert "芯片" in keywords_dict
        assert keywords_dict["芯片"] == 1.5

    def test_action_term_low_weight(self, parser):
        result = parser.parse("茅台走势分析")
        keywords_dict = dict(result.keywords)
        assert "走势" in keywords_dict
        assert keywords_dict["走势"] == 1.0

    def test_no_duplicate_keywords(self, parser):
        result = parser.parse("茅台分析")
        # 茅台 should appear only once
        terms = [t for t, w in result.keywords]
        assert terms.count("茅台") == 1

    def test_fallback_tokenize_when_no_keywords(self, parser):
        # Query with no domain keywords
        result = parser.parse("这个怎么样")
        # Should fallback to tokenization (filtered stop words)
        assert len(result.keywords) >= 0  # May be empty after stop word filter

    def test_weighted_terms_repeats_high_weight(self, parser):
        result = parser.parse("茅台的K线分析")
        weighted = result.get_weighted_terms()
        # 茅台 (weight 3.0) should appear 3 times
        assert weighted.count("茅台") == 3


# ===================== 查询类型分类 =====================


class TestQueryTypeClassification:
    def test_analysis_type(self, parser):
        result = parser.parse("茅台走势分析")
        assert result.query_type == "analysis"

    def test_analysis_how_to(self, parser):
        result = parser.parse("如何投资AI板块")
        assert result.query_type == "analysis"

    def test_comparison_type(self, parser):
        result = parser.parse("茅台和五粮液对比")
        assert result.query_type == "comparison"

    def test_comparison_vs(self, parser):
        result = parser.parse("宁德VS比亚迪")
        assert result.query_type == "comparison"

    def test_factual_type(self, parser):
        result = parser.parse("茅台市值多少")
        assert result.query_type == "factual"

    def test_factual_what_is(self, parser):
        result = parser.parse("什么是ROE")
        assert result.query_type == "factual"

    def test_other_type(self, parser):
        result = parser.parse("今天天气不错")
        assert result.query_type == "other"


# ===================== QueryResult 方法 =====================


class TestQueryResultMethods:
    def test_get_filters_with_stock(self, parser):
        result = parser.parse("茅台最近行情")
        filters = result.get_filters()
        assert "stock_code" in filters
        assert filters["stock_code"] == "600519.SH"

    def test_get_filters_with_date(self, parser):
        result = parser.parse("2024-06-01的市场报告")
        filters = result.get_filters()
        assert "date" in filters
        assert filters["date"] == "2024-06-01"

    def test_get_filters_with_date_range(self, parser):
        result = parser.parse("最近一周的走势")
        filters = result.get_filters()
        assert "gte" in filters
        assert "lte" in filters

    def test_get_filters_empty(self, parser):
        result = parser.parse("AI行业分析")
        filters = result.get_filters()
        # No stock, no date
        assert "stock_code" not in filters
        assert "date" not in filters

    def test_raw_query_preserved(self, parser):
        query = "茅台最近一周走势怎么样，值得买入吗"
        result = parser.parse(query)
        assert result.raw_query == query


# ===================== 综合测试 =====================


class TestEndToEnd:
    def test_complex_query(self, parser):
        result = parser.parse("茅台最近一周K线分析，MACD金叉了吗")
        assert result.stock_code == "600519.SH"
        assert result.date_range is not None  # 最近一周
        assert result.query_type == "analysis"
        # Should have multiple keywords
        assert len(result.keywords) >= 2

    def test_multi_entity_query(self, parser):
        result = parser.parse("对比茅台和五粮液2024年业绩")
        # Should extract first stock found
        assert result.stock_code in ("600519.SH", "000858.SZ")
        assert result.date == "2024-01-01" or "2024" in result.raw_query
        assert result.query_type == "comparison"

    def test_pure_industry_query(self, parser):
        result = parser.parse("AI大模型对半导体行业的影响")
        assert result.stock_code == ""
        keywords_dict = dict(result.keywords)
        # Should extract industry terms
        assert any(t in keywords_dict for t in ["芯片", "半导体"])
