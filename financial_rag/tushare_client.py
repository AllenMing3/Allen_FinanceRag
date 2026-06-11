"""
Tushare 数据客户端 — 获取 K 线和财务指标数据

功能:
- 股票 K 线: 日线/周线 via pro.daily() / pro.weekly()
- ETF K 线: via pro.fund_daily()
- 财务指标: via pro.fina_indicator()

数据源: Tushare Pro (https://tushare.pro)
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tushare 实例缓存
_ts_api = None

# 股票/ETF 全量列表缓存（避免每次搜索都拉取全量数据）
_stock_list_cache = {"data": None, "ts": 0.0}
_etf_list_cache = {"data": None, "ts": 0.0}
_CACHE_TTL = 3600  # 1 小时缓存有效期


def _get_api():
    """获取 Tushare Pro API 实例（懒初始化 + 缓存）"""
    global _ts_api
    if _ts_api is not None:
        return _ts_api

    from financial_rag.config import config
    token = config.tushare.token

    if not token or token == "your_tushare_token_here":
        logger.warning("Tushare token 未配置，请在 .env 中设置 TUSHARE_TOKEN")
        return None

    try:
        import tushare as ts
        ts.set_token(token)
        _ts_api = ts.pro_api()
        logger.info("Tushare API 初始化成功")
        return _ts_api
    except ImportError:
        logger.warning("请安装 tushare: pip install tushare")
        return None
    except Exception as e:
        logger.error(f"Tushare API 初始化失败: {e}")
        return None


# ===================== K 线数据 =====================

def fetch_stock_kline(
    ts_code: str,
    days: int = 30,
    period: str = "daily",
) -> pd.DataFrame:
    """
    获取股票 K 线数据

    Args:
        ts_code: Tushare 股票代码，如 '600519.SH'、'000858.SZ'
        days: 回溯天数
        period: 'daily' 或 'weekly'

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    api = _get_api()
    if api is None:
        return pd.DataFrame()

    end_date = datetime.now().strftime("%Y%m%d")
    # 多拉一些天数防节假日
    start_date = (datetime.now() - timedelta(days=days * 2 + 30)).strftime("%Y%m%d")

    try:
        if period == "weekly":
            df = api.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化列名
        df = df.rename(columns={
            "trade_date": "date",
            "vol": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(days)

        # 统一数值格式
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].round(3)

        return df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()

    except Exception as e:
        logger.error(f"获取股票 K 线失败 ({ts_code}): {e}")
        return pd.DataFrame()


def fetch_etf_kline(
    ts_code: str,
    days: int = 30,
) -> pd.DataFrame:
    """
    获取 ETF K 线数据

    Args:
        ts_code: ETF 代码，如 '510300.SH'、'159915.SZ'
        days: 回溯天数

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    api = _get_api()
    if api is None:
        return pd.DataFrame()

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days * 2 + 30)).strftime("%Y%m%d")

    try:
        df = api.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "trade_date": "date",
            "vol": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(days)

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].round(3)

        return df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()

    except Exception as e:
        logger.error(f"获取 ETF K 线失败 ({ts_code}): {e}")
        return pd.DataFrame()


# ===================== 财务指标 =====================

def fetch_financial_indicators(
    ts_code: str,
    periods: int = 4,
) -> List[Dict]:
    """
    获取财务指标数据

    Args:
        ts_code: 股票代码
        periods: 获取最近 N 个报告期

    Returns:
        List of dicts with key financial indicators
    """
    api = _get_api()
    if api is None:
        return []

    try:
        df = api.fina_indicator(ts_code=ts_code, limit=periods)
        if df is None or df.empty:
            return []

        results = []
        for _, row in df.head(periods).iterrows():
            indicator = {
                "period": str(row.get("end_date", "")),
                "eps": row.get("eps", None),
                "dt_eps": row.get("dt_eps", None),
                "roe": row.get("roe", None),
                "roe_waa": row.get("roe_waa", None),
                "grossprofit_margin": row.get("grossprofit_margin", None),
                "netprofit_margin": row.get("netprofit_margin", None),
                "debt_to_assets": row.get("debt_to_assets", None),
                "current_ratio": row.get("current_ratio", None),
                "quick_ratio": row.get("quick_ratio", None),
                "ocfps": row.get("ocfps", None),  # 每股经营现金流
                "bps": row.get("bps", None),  # 每股净资产
            }
            results.append(indicator)

        return results

    except Exception as e:
        logger.error(f"获取财务指标失败 ({ts_code}): {e}")
        return []


# ===================== 股票搜索 =====================

def search_stock(keyword: str, limit: int = 10) -> List[Dict]:
    """
    按关键词搜索股票

    Args:
        keyword: 股票名称关键词
        limit: 最大返回数

    Returns:
        [{"ts_code": "600519.SH", "name": "贵州茅台", "market": "主板", ...}, ...]
    """
    api = _get_api()
    if api is None:
        return []

    import time as _time

    try:
        # 使用缓存避免每次搜索都拉取全量股票列表
        now = _time.time()
        if _stock_list_cache["data"] is None or (now - _stock_list_cache["ts"]) > _CACHE_TTL:
            df = api.stock_basic(exchange="", list_status="L",
                                fields="ts_code,symbol,name,area,industry,market,list_date")
            if df is None or df.empty:
                return []
            _stock_list_cache["data"] = df
            _stock_list_cache["ts"] = now
        else:
            df = _stock_list_cache["data"]

        # 按名称过滤
        mask = df["name"].str.contains(keyword, na=False)
        matched = df[mask].head(limit)

        return matched.to_dict("records")

    except Exception as e:
        logger.error(f"搜索股票失败 ({keyword}): {e}")
        return []


def search_etf(keyword: str, limit: int = 10) -> List[Dict]:
    """
    按关键词搜索 ETF

    Args:
        keyword: ETF 名称关键词
        limit: 最大返回数

    Returns:
        [{"ts_code": "510300.SH", "name": "沪深300ETF", ...}, ...]
    """
    api = _get_api()
    if api is None:
        return []

    import time as _time

    try:
        now = _time.time()
        if _etf_list_cache["data"] is None or (now - _etf_list_cache["ts"]) > _CACHE_TTL:
            df = api.fund_basic(market="E", fields="ts_code,name,management,fund_type,list_date")
            if df is None or df.empty:
                return []
            _etf_list_cache["data"] = df
            _etf_list_cache["ts"] = now
        else:
            df = _etf_list_cache["data"]

        mask = df["name"].str.contains(keyword, na=False)
        matched = df[mask].head(limit)

        return matched.to_dict("records")

    except Exception as e:
        logger.error(f"搜索 ETF 失败 ({keyword}): {e}")
        return []


# ===================== 统计计算 =====================

def compute_kline_stats(df: pd.DataFrame) -> Dict:
    """
    对 K 线数据计算基础统计指标

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume

    Returns:
        Dict with key statistics
    """
    if df.empty or len(df) < 2:
        return {}

    closes = df["close"].values
    latest = float(closes[-1])
    period_high = float(df["high"].max())
    period_low = float(df["low"].min())
    first_close = float(closes[0])
    change_pct = round((latest - first_close) / first_close * 100, 2) if first_close else 0

    up_days = int((df["close"] > df["open"]).sum())
    down_days = int((df["close"] <= df["open"]).sum())
    avg_vol = int(df["volume"].mean()) if "volume" in df.columns else 0

    ma5 = round(float(pd.Series(closes).rolling(5).mean().iloc[-1]), 3) if len(closes) >= 5 else None
    ma10 = round(float(pd.Series(closes).rolling(10).mean().iloc[-1]), 3) if len(closes) >= 10 else None
    ma20 = round(float(pd.Series(closes).rolling(20).mean().iloc[-1]), 3) if len(closes) >= 20 else None

    return {
        "latest_close": latest,
        "period_high": period_high,
        "period_low": period_low,
        "period_change_pct": change_pct,
        "avg_volume": avg_vol,
        "up_days": up_days,
        "down_days": down_days,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
    }


def compute_technical_indicators(df: pd.DataFrame) -> Dict:
    """
    计算技术分析指标

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume

    Returns:
        Dict with MACD, RSI, Bollinger bands, etc.
    """
    if df.empty or len(df) < 26:
        return {}

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # MACD
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    # RSI (14 日)
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands (20 日)
    ma20 = pd.Series(closes).rolling(20).mean()
    std20 = pd.Series(closes).rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    # KDJ
    low_min = pd.Series(lows).rolling(9).min()
    high_max = pd.Series(highs).rolling(9).max()
    rsv = (pd.Series(closes) - low_min) / (high_max - low_min + 1e-10) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    return {
        "macd": {
            "dif": round(float(dif.iloc[-1]), 3),
            "dea": round(float(dea.iloc[-1]), 3),
            "macd": round(float(macd.iloc[-1]), 3),
            "signal": "金叉" if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2] else
                     "死叉" if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2] else
                     "多头" if dif.iloc[-1] > dea.iloc[-1] else "空头",
        },
        "rsi": {
            "value": round(float(rsi.iloc[-1]), 2),
            "signal": "超买" if rsi.iloc[-1] > 70 else "超卖" if rsi.iloc[-1] < 30 else "中性",
        },
        "bollinger": {
            "upper": round(float(upper.iloc[-1]), 3),
            "middle": round(float(ma20.iloc[-1]), 3),
            "lower": round(float(lower.iloc[-1]), 3),
            "position": "上轨附近" if closes[-1] > upper.iloc[-1] * 0.98 else
                       "下轨附近" if closes[-1] < lower.iloc[-1] * 1.02 else "中轨附近",
        },
        "kdj": {
            "k": round(float(k.iloc[-1]), 2),
            "d": round(float(d.iloc[-1]), 2),
            "j": round(float(j.iloc[-1]), 2),
            "signal": "金叉" if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] else
                     "死叉" if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] else
                     "多头" if k.iloc[-1] > d.iloc[-1] else "空头",
        },
    }
