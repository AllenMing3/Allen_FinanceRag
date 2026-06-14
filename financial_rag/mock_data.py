"""
Mock 数据生成器 — 为无 API Key / 无网络环境提供逼真的模拟数据

覆盖两个外部数据源:
1. Tushare: K 线、股票搜索、ETF 搜索、财务指标
2. 财经新闻: 新闻搜索、全量新闻（同花顺/新浪财经/东方财富）

所有 mock 函数返回与真实 API 完全相同的数据结构。
"""
import logging
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 固定随机种子，保证同一请求返回一致数据
random.seed(42)


# ============================================================================
#  Tushare Mock
# ============================================================================

# 模拟股票数据库
_MOCK_STOCKS = {
    "600519.SH": {"name": "贵州茅台", "industry": "白酒", "market": "主板", "base_price": 1680.0},
    "000858.SZ": {"name": "五粮液", "industry": "白酒", "market": "主板", "base_price": 145.0},
    "300750.SZ": {"name": "宁德时代", "industry": "电池", "market": "创业板", "base_price": 210.0},
    "002594.SZ": {"name": "比亚迪", "industry": "汽车", "market": "中小板", "base_price": 265.0},
    "600036.SH": {"name": "招商银行", "industry": "银行", "market": "主板", "base_price": 35.0},
    "601318.SH": {"name": "中国平安", "industry": "保险", "market": "主板", "base_price": 45.0},
    "000001.SZ": {"name": "平安银行", "industry": "银行", "market": "主板", "base_price": 11.5},
    "600900.SH": {"name": "长江电力", "industry": "电力", "market": "主板", "base_price": 28.0},
}

_MOCK_ETFS = {
    "510300.SH": {"name": "沪深300ETF", "fund_type": "股票型", "base_price": 3.85},
    "510500.SH": {"name": "中证500ETF", "fund_type": "股票型", "base_price": 5.62},
    "510050.SH": {"name": "上证50ETF", "fund_type": "股票型", "base_price": 2.78},
    "159915.SZ": {"name": "创业板ETF", "fund_type": "股票型", "base_price": 2.15},
    "159995.SZ": {"name": "芯片ETF", "fund_type": "股票型", "base_price": 1.25},
    "512880.SH": {"name": "证券ETF", "fund_type": "股票型", "base_price": 1.05},
    "515790.SH": {"name": "光伏ETF", "fund_type": "股票型", "base_price": 0.85},
}


def _generate_kline(base_price: float, days: int, volatility: float = 0.02) -> pd.DataFrame:
    """生成逼真的 K 线数据（几何布朗运动 + 日内波动）"""
    dates = pd.bdate_range(end=datetime.now(), periods=days, freq="B")

    # 几何布朗运动模拟价格
    np.random.seed(hash(str(base_price) + str(days)) % (2**31))
    returns = np.random.normal(0.0003, volatility, days)  # 微正漂移
    prices = base_price * np.exp(np.cumsum(returns))

    # 日内 OHLC
    opens = prices * (1 + np.random.uniform(-0.005, 0.005, days))
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.normal(0, 0.008, days)))
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.normal(0, 0.008, days)))
    closes = prices

    # 成交量（与价格波动正相关）
    base_vol = base_price * 100000
    vol_noise = np.random.lognormal(0, 0.3, days)
    volumes = (base_vol * vol_noise).astype(int)
    amounts = (volumes * closes).astype(int)

    df = pd.DataFrame({
        "date": dates[:days],
        "open": np.round(opens, 3),
        "high": np.round(highs, 3),
        "low": np.round(lows, 3),
        "close": np.round(closes, 3),
        "volume": volumes,
        "amount": amounts,
    })
    return df


def mock_stock_kline(ts_code: str, days: int = 30, period: str = "daily") -> pd.DataFrame:
    """模拟股票 K 线数据"""
    info = _MOCK_STOCKS.get(ts_code)
    if not info:
        # 未知代码 → 随机生成一个基准价格
        seed = int(hashlib.md5(ts_code.encode()).hexdigest()[:8], 16) % 1000
        base_price = 10 + seed * 0.5
    else:
        base_price = info["base_price"]

    actual_days = days if period == "daily" else days // 5
    vol = 0.02 if period == "daily" else 0.04
    df = _generate_kline(base_price, actual_days, volatility=vol)
    logger.info(f"[Mock] 股票 K 线: {ts_code}, {len(df)} 条 ({period})")
    return df


def mock_etf_kline(ts_code: str, days: int = 30) -> pd.DataFrame:
    """模拟 ETF K 线数据"""
    info = _MOCK_ETFS.get(ts_code)
    if not info:
        seed = int(hashlib.md5(ts_code.encode()).hexdigest()[:8], 16) % 100
        base_price = 0.5 + seed * 0.1
    else:
        base_price = info["base_price"]

    df = _generate_kline(base_price, days, volatility=0.015)
    logger.info(f"[Mock] ETF K 线: {ts_code}, {len(df)} 条")
    return df


def mock_search_stock(keyword: str, limit: int = 10) -> List[Dict]:
    """模拟股票搜索"""
    results = []
    for ts_code, info in _MOCK_STOCKS.items():
        if keyword in info["name"] or keyword in info.get("industry", ""):
            results.append({
                "ts_code": ts_code,
                "symbol": ts_code.split(".")[0],
                "name": info["name"],
                "area": "贵州" if "茅台" in info["name"] else "广东",
                "industry": info["industry"],
                "market": info["market"],
                "list_date": "20010101",
            })
        if len(results) >= limit:
            break

    # 如果没有匹配，返回一个通用结果
    if not results:
        results.append({
            "ts_code": "600000.SH",
            "symbol": "600000",
            "name": f"{keyword}相关",
            "area": "上海",
            "industry": "金融",
            "market": "主板",
            "list_date": "19991110",
        })

    logger.info(f"[Mock] 股票搜索 '{keyword}': {len(results)} 条")
    return results


def mock_search_etf(keyword: str, limit: int = 10) -> List[Dict]:
    """模拟 ETF 搜索"""
    results = []
    for ts_code, info in _MOCK_ETFS.items():
        if keyword in info["name"] or keyword in info.get("fund_type", ""):
            results.append({
                "ts_code": ts_code,
                "name": info["name"],
                "management": "华夏基金" if "510" in ts_code else "易方达",
                "fund_type": info["fund_type"],
                "list_date": "20120504",
            })
        if len(results) >= limit:
            break

    if not results:
        results.append({
            "ts_code": "510300.SH",
            "name": "沪深300ETF",
            "management": "华泰柏瑞",
            "fund_type": "股票型",
            "list_date": "20120504",
        })

    logger.info(f"[Mock] ETF 搜索 '{keyword}': {len(results)} 条")
    return results


def mock_financial_indicators(ts_code: str, periods: int = 4) -> List[Dict]:
    """模拟财务指标数据"""
    info = _MOCK_STOCKS.get(ts_code, {})
    base_price = info.get("base_price", 100.0)

    # 生成最近 4 个季度的报告日期
    now = datetime.now()
    report_dates = []
    for i in range(periods):
        q = (now.month - 1) // 3 - i
        year = now.year + q // 4
        quarter = q % 4
        end_month = (quarter + 1) * 3
        report_dates.append(f"{year}{end_month:02d}30")

    results = []
    for i, period in enumerate(report_dates):
        # 基于股票类型生成合理指标
        eps = round(base_price * random.uniform(0.02, 0.06) * (1 + i * 0.05), 2)
        roe = round(random.uniform(8.0, 35.0), 2)
        gross_margin = round(random.uniform(20.0, 92.0), 2)

        results.append({
            "period": period,
            "eps": eps,
            "dt_eps": round(eps * 0.95, 2),
            "roe": roe,
            "roe_waa": round(roe * 0.9, 2),
            "grossprofit_margin": gross_margin,
            "netprofit_margin": round(gross_margin * 0.6, 2),
            "debt_to_assets": round(random.uniform(20.0, 65.0), 2),
            "current_ratio": round(random.uniform(1.0, 3.5), 2),
            "quick_ratio": round(random.uniform(0.8, 2.5), 2),
            "ocfps": round(eps * random.uniform(0.8, 1.5), 2),
            "bps": round(base_price * random.uniform(0.2, 0.5), 2),
        })

    logger.info(f"[Mock] 财务指标: {ts_code}, {len(results)} 期")
    return results


# ============================================================================
#  RSS News Mock
# ============================================================================

_MOCK_NEWS_POOL = [
    {"title": "央行宣布降准0.5个百分点 释放长期资金约1万亿元", "source": "同花顺", "sentiment": "正面"},
    {"title": "贵州茅台2024年净利润同比增长15.38% 拟每股派息30.876元", "source": "新浪财经", "sentiment": "正面"},
    {"title": "比亚迪5月新能源汽车销量超33万辆 同比增长38%", "source": "东方财富", "sentiment": "正面"},
    {"title": "宁德时代获欧洲大单 动力电池出货量全球第一", "source": "同花顺", "sentiment": "正面"},
    {"title": "招商银行一季度营收同比增长5.2% 不良率降至0.91%", "source": "新浪财经", "sentiment": "正面"},
    {"title": "A股三大指数集体收涨 沪指站上3200点", "source": "东方财富", "sentiment": "正面"},
    {"title": "证监会发布深化科创板改革八条措施 支持硬科技企业上市", "source": "同花顺", "sentiment": "正面"},
    {"title": "人工智能概念股持续走强 多只个股涨停", "source": "东方财富", "sentiment": "正面"},
    {"title": "人民币汇率中间价报7.0988 较上日调升52个基点", "source": "新浪财经", "sentiment": "正面"},
    {"title": "国际油价突码80美元/桶 能源板块全线上涨", "source": "同花顺", "sentiment": "正面"},
    {"title": "中国平安一季度净利润同比增长22% 寿险改革成效显现", "source": "新浪财经", "sentiment": "正面"},
    {"title": "五粮液推出新品系列 加速高端化布局", "source": "东方财富", "sentiment": "正面"},
    {"title": "光伏产业链价格企稳 行业产能过剩问题有望缓解", "source": "同花顺", "sentiment": "正面"},
    {"title": "半导体设备国产替代加速 多家公司订单饱满", "source": "东方财富", "sentiment": "正面"},
    {"title": "新能源车渗透率突码50% 燃油车市场份额持续萎缩", "source": "新浪财经", "sentiment": "正面"},
    {"title": "多家银行下调存款利率 理财市场迎来资金回流", "source": "同花顺", "sentiment": "正面"},
    {"title": "消费复苏态势明确 5月社零总额同比增长6.8%", "source": "新浪财经", "sentiment": "正面"},
    {"title": "房地产政策持续优化 一线城市限购松绑效果显现", "source": "东方财富", "sentiment": "正面"},
    {"title": "锂电池技术突破 固态电池量产时间表提前", "source": "同花顺", "sentiment": "正面"},
    {"title": "医药板块估值修复 创新药企业业绩超预期", "source": "新浪财经", "sentiment": "正面"},
    {"title": "AI人工智能赋能金融科技 智能投顾市场规模快速扩张", "source": "东方财富", "sentiment": "正面"},
    {"title": "芯片行业景气度回升 AI芯片需求强劲拉动增长", "source": "同花顺", "sentiment": "正面"},
    {"title": "智能制造产业政策密集出台 工业机器人订单放量", "source": "新浪财经", "sentiment": "正面"},
    {"title": "云计算行业竞争加剧 头部企业市场份额持续提升", "source": "东方财富", "sentiment": "正面"},
    {"title": "中国经济复苏势头强劲 GDP增速超预期", "source": "同花顺", "sentiment": "正面"},
]


def mock_search_news(keyword: str, max_news: int = 30) -> Dict:
    """模拟新闻搜索 — 按关键词过滤并返回结构化结果"""
    import time
    t0 = time.time()

    # 从池中选出与关键词相关的新闻（宽松匹配）
    kw_parts = [
        kw.strip()
        for kw in keyword.replace("、", ",").replace("，", ",").split(",")
        if kw.strip()
    ]

    matched = []
    for item in _MOCK_NEWS_POOL:
        text = item["title"]
        # 宽松匹配：任一关键词子串命中即可
        if any(p in text for p in kw_parts if len(p) >= 2) or not kw_parts:
            matched.append(item)

    # 如果没匹配到，返回前几条通用新闻
    if not matched:
        matched = _MOCK_NEWS_POOL[:5]

    # 构造完整结构
    now = datetime.now()
    items = []
    for i, m in enumerate(matched[:max_news]):
        pub_time = (now - timedelta(hours=i * 2)).strftime("%Y-%m-%d %H:%M:%S")
        items.append({
            "title": m["title"],
            "content": m["title"] + "。详细分析内容..." + f"（模拟数据第{i+1}条）",
            "source": m["source"],
            "publish_time": pub_time,
            "url": f"https://example.com/news/{i+1}",
            "sentiment": m["sentiment"],
        })

    elapsed = (time.time() - t0) * 1000
    logger.info(f"[Mock] 新闻搜索 '{keyword}': {len(items)} 条")

    return {
        "keyword": keyword,
        "total": len(items),
        "items": items,
        "elapsed_ms": elapsed,
    }


def mock_fetch_all_news(max_per_source: int = 20) -> List[Dict]:
    """模拟获取全部新闻"""
    now = datetime.now()
    items = []
    for i, m in enumerate(_MOCK_NEWS_POOL[:max_per_source * 3]):
        pub_time = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        items.append({
            "title": m["title"],
            "content": m["title"] + "。详细分析内容...",
            "source": m["source"],
            "publish_time": pub_time,
            "url": f"https://example.com/news/{i+1}",
            "sentiment": m["sentiment"],
        })

    logger.info(f"[Mock] 全量新闻: {len(items)} 条")
    return items

