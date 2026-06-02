"""
ETF K线数据获取模块 — 基于 akshare 拉取 ETF 历史行情

功能:
- 搜索 ETF: 按关键词搜索相关 ETF（如 "人工智能"、"芯片"、"半导体"）
- K线数据: 获取指定 ETF 的日K/周K/月K 历史数据
- 数据源: 新浪财经 (通过 akshare fund_etf_hist_sina)

使用示例:
    from financial_rag.etf_fetcher import search_etf, fetch_etf_kline

    # 搜索 AI 相关 ETF
    results = search_etf("人工智能")
    # -> [{"code": "sz159819", "name": "人工智能ETF易方达", ...}, ...]

    # 拉取 K 线
    df = fetch_etf_kline("sz159819", days=30)
    # -> DataFrame with date/open/high/low/close/volume/amount
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ETF_CACHE: Optional[pd.DataFrame] = None


def _get_all_etfs() -> pd.DataFrame:
    """获取全市场 ETF 列表（带缓存）"""
    global _ETF_CACHE
    if _ETF_CACHE is not None:
        return _ETF_CACHE
    import akshare as ak
    _ETF_CACHE = ak.fund_etf_category_sina(symbol="ETF基金")
    return _ETF_CACHE


def search_etf(keyword: str, limit: int = 10) -> List[Dict]:
    """
    按关键词搜索 ETF
    
    Args:
        keyword: 搜索关键词，如 "人工智能"、"芯片"、"半导体"、"5G"
        limit: 最多返回条数
    
    Returns:
        [{"code": "sz159819", "name": "人工智能ETF易方达", "price": ..., "change_pct": ...}, ...]
    """
    df = _get_all_etfs()
    if df is None or df.empty:
        return []

    mask = df["名称"].str.contains(keyword, na=False)
    matched = df[mask].head(limit)

    results = []
    for _, row in matched.iterrows():
        results.append({
            "code": row["代码"],
            "name": row["名称"],
            "price": row.get("最新价", None),
            "change_pct": row.get("涨跌幅", None),
            "volume": row.get("成交量", None),
            "amount": row.get("成交额", None),
        })
    return results


def fetch_etf_kline(
    symbol: str,
    days: int = 30,
    period: str = "daily",
) -> pd.DataFrame:
    """
    获取 ETF 历史 K 线数据
    
    Args:
        symbol: ETF 代码，如 "sz159819"（深交所）或 "sh515070"（上交所）
        days: 回溯天数
        period: K线周期，目前仅支持 "daily"
    
    Returns:
        DataFrame, columns: date, open, high, low, close, volume, amount
    """
    import akshare as ak

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")  # 多拉一点防节假日

    try:
        df = ak.fund_etf_hist_sina(symbol=symbol)
    except Exception as e:
        logger.error(f"获取 ETF K线失败 ({symbol}): {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 过滤日期范围
    df["date"] = pd.to_datetime(df["date"])
    start_dt = pd.to_datetime(start_date)
    df = df[df["date"] >= start_dt].copy()
    df = df.tail(days)  # 精确取最近 N 个交易日

    # 统一数值格式
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].round(3)

    return df.reset_index(drop=True)


def compute_kline_stats(df: pd.DataFrame) -> Dict:
    """
    对 K 线数据计算基础统计指标
    
    Returns:
        {
            "latest_close": 最新收盘价,
            "period_high": 区间最高,
            "period_low": 区间最低,
            "period_change_pct": 区间涨跌幅(%),
            "avg_volume": 平均成交量,
            "up_days": 上涨天数,
            "down_days": 下跌天数,
            "ma5": 5日均线,
            "ma10": 10日均线,
        }
    """
    if df.empty or len(df) < 2:
        return {}

    closes = df["close"].values
    latest = closes[-1]
    period_high = float(df["high"].max())
    period_low = float(df["low"].min())
    first_close = closes[0]
    change_pct = round((latest - first_close) / first_close * 100, 2) if first_close else 0

    up_days = int((df["close"] > df["open"]).sum())
    down_days = int((df["close"] <= df["open"]).sum())
    avg_vol = int(df["volume"].mean()) if "volume" in df.columns else 0

    ma5 = round(float(pd.Series(closes).rolling(5).mean().iloc[-1]), 3) if len(closes) >= 5 else None
    ma10 = round(float(pd.Series(closes).rolling(10).mean().iloc[-1]), 3) if len(closes) >= 10 else None

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
    }
