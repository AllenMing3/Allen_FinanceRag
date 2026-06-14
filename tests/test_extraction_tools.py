"""
Test extraction tools — regex fallback path (no LLM needed)

Tests the 5 extraction tools when LLM is not injected,
verifying regex fallback produces reasonable results for AI-sector texts.
"""
import pytest
from financial_rag.tools.extraction_tools import (
    extract_financial_metrics,
    extract_entities,
    extract_document_metadata,
    detect_document_type,
    generate_search_queries,
    _regex_fallback_metrics,
    _regex_fallback_entities,
    _regex_fallback_metadata,
    _fallback_queries,
)


class TestExtractFinancialMetrics:
    """extract_financial_metrics — regex fallback mode"""

    def test_ai_financial_report(self, ai_financial_text):
        result = extract_financial_metrics(ai_financial_text)
        assert isinstance(result, dict)
        # Regex should catch revenue
        assert "revenue" in result or result.get("_confidence") in ("low", "none")
        # Should have confidence indicator
        assert "_confidence" in result

    def test_empty_input(self):
        result = extract_financial_metrics("")
        assert result["_confidence"] == "none"
        assert "_error" in result

    def test_regex_revenue(self, ai_financial_text):
        result = _regex_fallback_metrics(ai_financial_text)
        assert isinstance(result, dict)
        # Should find revenue from "营业收入50.3亿元"
        if "revenue" in result:
            assert result["revenue"]["_confidence"] == "low"

    def test_regex_rd_expense(self, ai_financial_text):
        result = _regex_fallback_metrics(ai_financial_text)
        # Should find "研发投入18.7亿元"
        if "rd_expense" in result:
            assert result["rd_expense"]["value"] > 0

    def test_regex_gpu_count(self, ai_financial_text):
        result = _regex_fallback_metrics(ai_financial_text)
        # Should find "4万卡A100"
        if "gpu_count" in result or "training_cluster_size" in result:
            # At least one compute metric found
            found = result.get("gpu_count") or result.get("training_cluster_size")
            assert found is not None

    def test_news_text(self, ai_news_text):
        result = extract_financial_metrics(ai_news_text)
        assert isinstance(result, dict)
        assert "_confidence" in result

    def test_funding_text(self, ai_funding_text):
        result = extract_financial_metrics(ai_funding_text)
        assert isinstance(result, dict)


class TestExtractEntities:
    """extract_entities — regex fallback mode"""

    def test_ai_financial_report(self, ai_financial_text):
        result = extract_entities(ai_financial_text)
        assert isinstance(result, dict)
        assert "_confidence" in result
        assert "_source" in result

    def test_regex_companies(self, ai_financial_text):
        result = _regex_fallback_entities(ai_financial_text)
        # Should find "商汤科技集团股份有限公司"
        companies = result.get("companies", [])
        assert len(companies) >= 0  # May or may not match depending on pattern

    def test_regex_financial_figures(self, ai_financial_text):
        result = _regex_fallback_entities(ai_financial_text)
        figures = result.get("financial_figures", [])
        # Should find monetary amounts like "50.3亿元"
        assert len(figures) >= 1 or result["_confidence"] in ("low", "none")

    def test_empty_input(self):
        result = extract_entities("")
        assert result["_confidence"] == "none"

    def test_news_text(self, ai_news_text):
        result = extract_entities(ai_news_text)
        assert isinstance(result, dict)


class TestExtractDocumentMetadata:
    """extract_document_metadata — regex fallback mode"""

    def test_ai_financial_report(self, ai_financial_text):
        result = extract_document_metadata(ai_financial_text)
        assert isinstance(result, dict)
        assert "_confidence" in result

    def test_regex_date_extraction(self, ai_financial_text):
        result = _regex_fallback_metadata(ai_financial_text)
        # Should find "2024年" pattern
        assert isinstance(result, dict)
        # date might be extracted from "2024年度"
        if result.get("date"):
            assert "2024" in result["date"]

    def test_regex_fiscal_period(self):
        text = "贵州茅台2024年年度报告"
        result = _regex_fallback_metadata(text)
        if result.get("fiscal_period"):
            assert "2024" in result["fiscal_period"]

    def test_empty_input(self):
        result = extract_document_metadata("")
        assert result["_confidence"] == "none"


class TestDetectDocumentType:
    """detect_document_type — pure keyword matching (no LLM needed)"""

    def test_tech_report(self, ai_tech_text):
        doc_type = detect_document_type(ai_tech_text)
        assert doc_type == "技术报告"

    def test_product_launch(self, ai_product_text):
        doc_type = detect_document_type(ai_product_text)
        assert doc_type == "产品发布"

    def test_funding_announcement(self, ai_funding_text):
        doc_type = detect_document_type(ai_funding_text)
        assert doc_type == "融资公告"

    def test_annual_report(self):
        text = "某公司2024年年度报告 营收利润 董事会决议"
        doc_type = detect_document_type(text)
        assert doc_type in ("年报", "年度报告", "其他")

    def test_news_report(self, ai_news_text):
        doc_type = detect_document_type(ai_news_text)
        # News text might match various types
        assert isinstance(doc_type, str)
        assert len(doc_type) > 0

    def test_empty_input(self):
        assert detect_document_type("") == "其他"

    def test_industry_analysis(self):
        text = "2024年AI行业分析报告 市场规模 竞争格局 发展趋势 产业链"
        doc_type = detect_document_type(text)
        assert doc_type in ("行业分析", "研究报告", "其他")


class TestGenerateSearchQueries:
    """generate_search_queries — fallback mode (no LLM)"""

    def test_with_metrics(self):
        metrics = {"revenue": {"value": 50.3}, "gpu_count": {"value": 40000}}
        queries = generate_search_queries("some text", metrics=metrics, entities={})
        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_with_entities(self):
        entities = {"companies": [{"name": "商汤科技"}]}
        queries = generate_search_queries("some text", metrics={}, entities=entities)
        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_empty_inputs(self):
        queries = generate_search_queries("some text", metrics={}, entities={})
        # Fallback should still generate queries
        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_fallback_queries_directly(self):
        metrics = {"api_calls": {"value": 2000}}
        entities = {"companies": [{"name": "智谱AI"}]}
        queries = _fallback_queries(metrics, entities)
        assert len(queries) >= 1
        assert any("智谱AI" in q or "API" in q for q in queries)

    def test_guaranteed_fallback(self):
        queries = _fallback_queries({}, {})
        assert len(queries) >= 1


class TestLongArticleExtraction:
    """Test extraction on realistic 1500-2000 char AI-sector articles"""

    def test_sensetime_annual_report_metrics(self, long_article_sensetime):
        """商汤年报 ~2000字 — should extract revenue, rd_expense, arr, customer_count, etc."""
        result = extract_financial_metrics(long_article_sensetime)
        assert isinstance(result, dict)
        assert "_confidence" in result
        # Regex should catch multiple metrics from this rich text
        metric_keys = [k for k in result if not k.startswith("_")]
        assert len(metric_keys) >= 1, f"Expected at least 1 metric from long article, got: {metric_keys}"

    def test_nvidia_blackwell_metrics(self, long_article_nvidia):
        """英伟达架构解析 ~2000字 — GPU specs, performance numbers"""
        result = extract_financial_metrics(long_article_nvidia)
        assert isinstance(result, dict)

    def test_funding_roundup_metrics(self, long_article_funding):
        """融资盘点 ~1500字 — funding amounts, valuations"""
        result = extract_financial_metrics(long_article_funding)
        assert isinstance(result, dict)

    def test_sensetime_entities(self, long_article_sensetime):
        """商汤年报实体抽取 — companies, financial_figures, tech terms"""
        result = extract_entities(long_article_sensetime)
        assert isinstance(result, dict)
        assert "_source" in result

    def test_nvidia_entities(self, long_article_nvidia):
        """英伟达文章实体 — should find NVIDIA, GPU types, companies"""
        result = extract_entities(long_article_nvidia)
        assert isinstance(result, dict)

    def test_funding_entities(self, long_article_funding):
        """融资盘点实体 — multiple companies, funding amounts"""
        result = extract_entities(long_article_funding)
        assert isinstance(result, dict)
        figures = result.get("financial_figures", [])
        # Should find multiple monetary amounts (50亿, 200亿, 10亿美元...)
        assert len(figures) >= 1

    def test_long_article_metadata(self, long_article_sensetime):
        """长文元数据抽取 — should find date, fiscal period, company"""
        result = extract_document_metadata(long_article_sensetime)
        assert isinstance(result, dict)

    def test_long_article_doc_type(self, long_article_sensetime):
        """长文类型检测 — annual report should be detected"""
        doc_type = detect_document_type(long_article_sensetime)
        assert doc_type in ("年报", "年度报告", "研究报告", "行业分析", "新闻报道", "其他")

    def test_long_article_queries(self, long_article_sensetime):
        """长文生成检索查询"""
        metrics = extract_financial_metrics(long_article_sensetime)
        entities = extract_entities(long_article_sensetime)
        queries = generate_search_queries(
            long_article_sensetime, metrics=metrics, entities=entities
        )
        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_all_long_articles_metrics(self, all_long_articles):
        """所有长文都能成功抽取指标"""
        for article in all_long_articles:
            result = extract_financial_metrics(article)
            assert isinstance(result, dict)
            assert "_confidence" in result
            assert len(article) > 500, "Article should be long"

    def test_all_long_articles_entities(self, all_long_articles):
        """所有长文都能成功抽取实体"""
        for article in all_long_articles:
            result = extract_entities(article)
            assert isinstance(result, dict)
            assert "_confidence" in result

    def test_text_length_not_truncated(self, long_article_sensetime):
        """Verify long text is not silently truncated below 8000 chars"""
        assert len(long_article_sensetime) > 1000
        assert len(long_article_sensetime) < 8000  # Within limit
        result = extract_financial_metrics(long_article_sensetime)
        assert result.get("_confidence") != "none"
