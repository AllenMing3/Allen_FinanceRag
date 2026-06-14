"""
Test mock data generators — verify mock data matches real API structures
"""
import pytest
from financial_rag.mock_data import (
    mock_stock_kline,
    mock_etf_kline,
    mock_search_stock,
    mock_search_etf,
    mock_financial_indicators,
    mock_search_news,
    mock_fetch_all_news,
    mock_long_article,
    MOCK_LONG_ARTICLES,
)


class TestMockKline:
    def test_stock_kline_shape(self):
        df = mock_stock_kline("600519.SH", days=30)
        assert len(df) == 30
        assert set(df.columns) == {"date", "open", "high", "low", "close", "volume", "amount"}

    def test_stock_kline_values(self):
        df = mock_stock_kline("600519.SH", days=30)
        assert (df["high"] >= df["low"]).all()
        assert (df["volume"] > 0).all()
        assert (df["close"] > 0).all()

    def test_stock_kline_weekly(self):
        df = mock_stock_kline("600519.SH", days=30, period="weekly")
        assert len(df) == 6  # 30 / 5

    def test_etf_kline_shape(self):
        df = mock_etf_kline("510300.SH", days=30)
        assert len(df) == 30
        assert "close" in df.columns

    def test_unknown_stock(self):
        df = mock_stock_kline("999999.SH", days=10)
        assert len(df) == 10
        assert (df["close"] > 0).all()

    def test_unknown_etf(self):
        df = mock_etf_kline("999999.SH", days=10)
        assert len(df) == 10


class TestMockSearch:
    def test_search_stock_found(self):
        results = mock_search_stock("茅台")
        assert len(results) >= 1
        assert results[0]["ts_code"] == "600519.SH"
        assert results[0]["name"] == "贵州茅台"

    def test_search_stock_not_found(self):
        results = mock_search_stock("不存在的公司")
        assert len(results) >= 1  # Returns fallback

    def test_search_etf_found(self):
        results = mock_search_etf("沪深300")
        assert len(results) >= 1
        assert "510300" in results[0]["ts_code"]

    def test_search_etf_not_found(self):
        results = mock_search_etf("不存在的ETF")
        assert len(results) >= 1


class TestMockFinancialIndicators:
    def test_indicator_structure(self):
        results = mock_financial_indicators("600519.SH", periods=4)
        assert len(results) == 4
        first = results[0]
        assert "period" in first
        assert "eps" in first
        assert "roe" in first
        assert "grossprofit_margin" in first

    def test_indicator_values(self):
        results = mock_financial_indicators("600519.SH", periods=4)
        for r in results:
            assert r["roe"] > 0
            assert r["grossprofit_margin"] > 0
            assert r["current_ratio"] > 0


class TestMockNews:
    def test_search_news_structure(self):
        result = mock_search_news("AI")
        assert "keyword" in result
        assert "total" in result
        assert "items" in result
        assert result["total"] >= 1

    def test_search_news_items(self):
        result = mock_search_news("AI")
        for item in result["items"]:
            assert "title" in item
            assert "content" in item
            assert "source" in item
            assert "publish_time" in item
            assert "url" in item

    def test_search_news_ai_sector(self):
        """News pool should be AI-sector focused"""
        result = mock_search_news("AI")
        titles = [item["title"] for item in result["items"]]
        # At least some should be AI-related
        ai_count = sum(1 for t in titles if any(
            kw in t for kw in ["AI", "大模型", "GPU", "芯片", "算力", "模型"]
        ))
        assert ai_count >= 1

    def test_fetch_all_news(self):
        items = mock_fetch_all_news()
        assert len(items) >= 1
        assert "title" in items[0]
        assert "content" in items[0]

    def test_news_no_traditional_finance(self):
        """Verify no 茅台/央行 in news pool"""
        result = mock_search_news("")
        all_text = " ".join(item["title"] for item in result["items"])
        assert "茅台" not in all_text
        assert "央行" not in all_text
        assert "降准" not in all_text


class TestMockLongArticles:
    def test_long_article_count(self):
        assert len(MOCK_LONG_ARTICLES) >= 3

    def test_article_length(self):
        """Each article should be 800+ chars"""
        for article in MOCK_LONG_ARTICLES:
            assert len(article["content"]) > 800, f"Article '{article['title'][:20]}' too short: {len(article['content'])} chars"

    def test_article_structure(self):
        for article in MOCK_LONG_ARTICLES:
            assert "title" in article
            assert "content" in article
            assert "source" in article

    def test_mock_long_article_function(self):
        result = mock_long_article(0)
        assert "title" in result
        assert "content" in result
        assert len(result["content"]) > 1000
        assert result["text_length"] > 1000

    def test_mock_long_article_invalid_index(self):
        result = mock_long_article(999)
        assert len(result["content"]) > 1000  # Falls back to index 0

    def test_articles_ai_sector(self):
        """All articles should be AI-sector focused"""
        all_text = " ".join(a["content"] for a in MOCK_LONG_ARTICLES)
        ai_keywords = ["AI", "大模型", "GPU", "算力", "模型", "芯片"]
        ai_count = sum(1 for kw in ai_keywords if kw in all_text)
        assert ai_count >= 3, "Articles should be heavily AI-focused"

    def test_no_traditional_finance_in_articles(self):
        all_text = " ".join(a["content"] for a in MOCK_LONG_ARTICLES)
        assert "茅台" not in all_text
        assert "五粮液" not in all_text
