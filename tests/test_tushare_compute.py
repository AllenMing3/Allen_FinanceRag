"""
Test tushare_client compute functions — compute_kline_stats, compute_technical_indicators

These are pure computation functions (no API calls, no LLM) — ideal for unit testing.
Also covers analyze_kline from kline_tools.py (wires fetch + stats + indicators).
"""
import math
import pytest
import pandas as pd

from financial_rag.tushare_client import compute_kline_stats, compute_technical_indicators


# ===================== Helpers =====================


def _make_kline_df(n: int = 60, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic K-line DataFrame with n trading days."""
    import random
    rng = random.Random(seed)
    rows = []
    price = start_price
    for i in range(n):
        change = rng.gauss(0, 0.02)  # ~2% daily vol
        o = round(price * (1 + rng.gauss(0, 0.005)), 2)
        c = round(price * (1 + change), 2)
        h = round(max(o, c) * (1 + abs(rng.gauss(0, 0.005))), 2)
        low = round(min(o, c) * (1 - abs(rng.gauss(0, 0.005))), 2)
        vol = rng.randint(100_000, 10_000_000)
        rows.append({
            "date": f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}",
            "open": o, "high": h, "low": low, "close": c,
            "volume": vol, "amount": round(vol * c, 2),
        })
        price = c
    return pd.DataFrame(rows)


# ===================== compute_kline_stats =====================


class TestComputeKlineStats:

    def test_basic_shape(self):
        df = _make_kline_df(30)
        stats = compute_kline_stats(df)
        assert "latest_close" in stats
        assert "period_high" in stats
        assert "period_low" in stats
        assert "period_change_pct" in stats
        assert "up_days" in stats
        assert "down_days" in stats

    def test_latest_close_matches_last_row(self):
        df = _make_kline_df(20)
        stats = compute_kline_stats(df)
        assert stats["latest_close"] == float(df["close"].iloc[-1])

    def test_period_high_low(self):
        df = _make_kline_df(30)
        stats = compute_kline_stats(df)
        assert stats["period_high"] == float(df["high"].max())
        assert stats["period_low"] == float(df["low"].min())

    def test_change_pct_calculation(self):
        df = _make_kline_df(10, start_price=100.0)
        stats = compute_kline_stats(df)
        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        expected = round((last_close - first_close) / first_close * 100, 2)
        assert stats["period_change_pct"] == expected

    def test_up_down_days_sum(self):
        df = _make_kline_df(30)
        stats = compute_kline_stats(df)
        assert stats["up_days"] + stats["down_days"] == 30

    def test_ma_values(self):
        df = _make_kline_df(30)
        stats = compute_kline_stats(df)
        # MA5 should be present (30 >= 5)
        assert stats["ma5"] is not None
        # MA10 should be present (30 >= 10)
        assert stats["ma10"] is not None
        # MA20 should be present (30 >= 20)
        assert stats["ma20"] is not None

    def test_ma_none_when_insufficient_data(self):
        df = _make_kline_df(3)
        stats = compute_kline_stats(df)
        assert stats["ma5"] is None
        assert stats["ma10"] is None
        assert stats["ma20"] is None

    def test_empty_df(self):
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        stats = compute_kline_stats(df)
        assert stats == {}

    def test_single_row(self):
        df = _make_kline_df(1)
        stats = compute_kline_stats(df)
        assert stats == {}

    def test_avg_volume(self):
        df = _make_kline_df(10)
        stats = compute_kline_stats(df)
        assert stats["avg_volume"] == int(df["volume"].mean())


# ===================== compute_technical_indicators =====================


class TestComputeTechnicalIndicators:

    def test_returns_empty_for_short_df(self):
        df = _make_kline_df(10)
        indicators = compute_technical_indicators(df)
        assert indicators == {}  # need >= 26 rows

    def test_full_indicator_set(self):
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        assert "macd" in indicators
        assert "rsi" in indicators
        assert "bollinger" in indicators
        assert "kdj" in indicators

    def test_macd_structure(self):
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        macd = indicators["macd"]
        assert "dif" in macd
        assert "dea" in macd
        assert "macd" in macd
        assert "signal" in macd
        assert macd["signal"] in ("金叉", "死叉", "多头", "空头")

    def test_macd_values_are_numeric(self):
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        macd = indicators["macd"]
        assert isinstance(macd["dif"], float)
        assert isinstance(macd["dea"], float)
        assert isinstance(macd["macd"], float)

    def test_rsi_range(self):
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        rsi = indicators["rsi"]
        assert 0 <= rsi["value"] <= 100
        assert rsi["signal"] in ("超买", "超卖", "中性")

    def test_rsi_signal_thresholds(self):
        """RSI > 70 = 超买, < 30 = 超卖, else 中性"""
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        rsi_val = indicators["rsi"]["value"]
        signal = indicators["rsi"]["signal"]
        if rsi_val > 70:
            assert signal == "超买"
        elif rsi_val < 30:
            assert signal == "超卖"
        else:
            assert signal == "中性"

    def test_bollinger_structure(self):
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        boll = indicators["bollinger"]
        assert "upper" in boll
        assert "middle" in boll
        assert "lower" in boll
        assert "position" in boll
        # Upper > Middle > Lower
        assert boll["upper"] > boll["middle"] > boll["lower"]
        assert boll["position"] in ("上轨附近", "下轨附近", "中轨附近")

    def test_kdj_structure(self):
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        kdj = indicators["kdj"]
        assert "k" in kdj
        assert "d" in kdj
        assert "j" in kdj
        assert "signal" in kdj
        assert kdj["signal"] in ("金叉", "死叉", "多头", "空头")

    def test_kdj_j_formula(self):
        """J = 3K - 2D"""
        df = _make_kline_df(60)
        indicators = compute_technical_indicators(df)
        kdj = indicators["kdj"]
        expected_j = round(3 * kdj["k"] - 2 * kdj["d"], 2)
        assert abs(kdj["j"] - expected_j) < 0.02  # float tolerance

    def test_exactly_26_rows(self):
        """Boundary: minimum rows for indicators"""
        df = _make_kline_df(26)
        indicators = compute_technical_indicators(df)
        assert "macd" in indicators
        assert "rsi" in indicators

    def test_25_rows_returns_empty(self):
        """Just below threshold"""
        df = _make_kline_df(25)
        indicators = compute_technical_indicators(df)
        assert indicators == {}


# ===================== analyze_kline integration =====================


class TestAnalyzeKline:

    def test_analyze_with_mock_data(self):
        """analyze_kline should work end-to-end with mock mode"""
        from unittest.mock import patch
        from financial_rag.tools.kline_tools import analyze_kline

        mock_df = _make_kline_df(60)
        with patch("financial_rag.tushare_client.fetch_stock_kline", return_value=mock_df), \
             patch("financial_rag.tushare_client.fetch_etf_kline", return_value=mock_df):
            result = analyze_kline(ts_code="600519.SH", days=60)
            assert result["ts_code"] == "600519.SH"
            assert result["data_points"] == 60
            assert result["is_etf"] is False
            assert "stats" in result
            assert "indicators" in result
            assert "latest_close" in result["stats"]
            assert "macd" in result["indicators"]

    def test_analyze_etf_detection(self):
        """ETF codes starting with 51 or 159 should set is_etf=True"""
        from unittest.mock import patch
        from financial_rag.tools.kline_tools import analyze_kline

        mock_df = _make_kline_df(30)
        with patch("financial_rag.tushare_client.fetch_etf_kline", return_value=mock_df):
            result = analyze_kline(ts_code="510300.SH", days=30)
            assert result["is_etf"] is True

    def test_analyze_empty_df_returns_error(self):
        """Empty DataFrame should return error dict"""
        from unittest.mock import patch
        from financial_rag.tools.kline_tools import analyze_kline

        empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        with patch("financial_rag.tushare_client.fetch_stock_kline", return_value=empty_df):
            result = analyze_kline(ts_code="999999.SH", days=30)
            assert "error" in result
