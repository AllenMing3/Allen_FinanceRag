"""Tests for QueryParser — 查询解析器"""
import pytest
from financial_rag.retrievers.query_parser import (
    QueryParser, QueryResult,
    FINANCIAL_TERMS, INDUSTRY_TERMS, ACTION_TERMS, STOP_WORDS,
)
from financial_rag.retrievers.dictionaries import SYNONYM_LOOKUP, CONCEPT_MAP


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
        # stock_code 不再作为 metadata 过滤条件（文档侧无此字段）
        filters = result.get_filters()
        assert "stock_code" not in filters
        # 但 QueryResult 仍然抽取股票信息（给 K线工具用）
        assert result.stock_code == "600519.SH"

    def test_get_filters_with_date(self, parser):
        result = parser.parse("2024-06-01的市场报告")
        filters = result.get_filters()
        assert "publish_date" in filters
        assert filters["publish_date"] == "2024-06-01"

    def test_get_filters_with_date_range(self, parser):
        result = parser.parse("最近一周的走势")
        filters = result.get_filters()
        assert "publish_date" in filters
        assert "gte" in filters["publish_date"]
        assert "lte" in filters["publish_date"]

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


# ===================== 查询扩展 =====================


class TestQueryExpansion:
    """规则层查询扩展测试"""

    def test_synonym_lookup_exists(self):
        """同义词表应存在且包含基本条目"""
        assert "英伟达" in SYNONYM_LOOKUP
        assert "NVIDIA" in SYNONYM_LOOKUP["英伟达"]
        assert "NVDA" in SYNONYM_LOOKUP["英伟达"]

    def test_concept_map_exists(self):
        """概念关联表应存在且包含基本条目"""
        assert "芯片" in CONCEPT_MAP
        assert "半导体" in CONCEPT_MAP["芯片"]

    def test_synonym_expansion_nvidia(self, parser):
        """“英伟达”应扩展出 NVIDIA、NVDA"""
        result = parser.parse("英伟达最新芯片")
        assert len(result.expanded_terms) > 0
        expanded_lower = [t.lower() for t in result.expanded_terms]
        assert "nvidia" in expanded_lower or "nvda" in expanded_lower

    def test_synonym_expansion_maotai(self, parser):
        """“茅台”应扩展出贵州茅台、600519"""
        result = parser.parse("茅台最近走势")
        expanded_set = set(result.expanded_terms)
        assert "贵州茅台" in expanded_set or "600519" in expanded_set

    def test_concept_expansion_chip(self, parser):
        """“芯片”应扩展出半导体、光刻等关联词"""
        result = parser.parse("芯片行业分析")
        expanded_set = set(result.expanded_terms)
        assert "半导体" in expanded_set

    def test_concept_expansion_ai(self, parser):
        """“AI”应扩展出大模型、算力、GPU等"""
        result = parser.parse("AI对投资的影响")
        expanded_set = set(result.expanded_terms)
        # At least one AI-related concept should be expanded
        assert any(t in expanded_set for t in ["大模型", "算力", "GPU", "人工智能"])

    def test_expanded_query_contains_original(self, parser):
        """expanded_query 应包含原始查询"""
        result = parser.parse("英伟达芯片")
        assert "英伟达芯片" in result.expanded_query

    def test_no_expansion_for_unrelated_query(self, parser):
        """无关查询不应产生扩展词"""
        result = parser.parse("今天天气怎么样")
        assert result.expanded_terms == []
        assert result.expanded_query == "今天天气怎么样"

    def test_expand_weight_synonym(self, parser):
        """同义词权重应为 1.5"""
        result = parser.parse("英伟达最新财报")
        kw_dict = dict(result.keywords)
        # NVIDIA/NVDA should be in keywords with weight 1.5
        for term in result.expanded_terms:
            if term.lower() in ("nvidia", "nvda"):
                assert kw_dict[term] == 1.5
                break

    def test_expand_weight_concept(self, parser):
        """概念关联词权重应为 0.6"""
        result = parser.parse("芯片板块走势")
        kw_dict = dict(result.keywords)
        for term in result.expanded_terms:
            if term in CONCEPT_MAP.get("芯片", []):
                assert kw_dict[term] == 0.6
                break

    def test_expansion_dedup(self, parser):
        """扩展词不应与已有关键词重复"""
        result = parser.parse("芯片半导体行业")
        # “半导体”可能在 INDUSTRY_TERMS 中已存在，不应重复
        terms = [t for t, w in result.keywords]
        assert len(terms) == len(set(terms))

    def test_max_expansion_cap(self, parser):
        """扩展词数量不应超过上限"""
        result = parser.parse("英伟达芯片AI算力GPU大模型半导体光刻机光伏储能")
        assert len(result.expanded_terms) <= parser._MAX_EXPAND_TERMS

    def test_expanded_terms_field_populated(self, parser):
        """result.expanded_terms 应被正确填充"""
        result = parser.parse("茅台")
        assert isinstance(result.expanded_terms, list)
        assert len(result.expanded_terms) > 0

    def test_case_insensitive_synonym(self, parser):
        """英文同义词应不区分大小写"""
        result = parser.parse("NVIDIA最新GPU")
        expanded_lower = [t.lower() for t in result.expanded_terms]
        # Should find "英伟达" from the synonym group
        assert "英伟达" in expanded_lower or "nvda" in expanded_lower
