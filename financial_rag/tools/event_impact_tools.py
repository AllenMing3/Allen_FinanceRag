"""
事件影响分析工具 — 将新闻事件与 K 线数据做映射

V1 能力:
- fetch_date_events: 获取某日发生的事件/新闻
- fetch_kline_context: 获取某日前后的 K 线数据
- assess_event_impact: LLM 分类事件影响（利好/利空/中性 + 影响因子）

数据源: 同花顺/新浪/东方财富 (新闻), Tushare (K 线)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


# ===================== 工具 1: 获取某日事件 =====================

def fetch_date_events(
    date: str = "",
    keyword: str = "",
    max_events: int = 20,
) -> Dict:
    """获取指定日期发生的重要事件/新闻。

    从国内财经 API (同花顺/新浪/东方财富) 拉取新闻，
    按日期过滤，返回当日事件列表。

    Args:
        date: 日期字符串，格式 'YYYY-MM-DD'，默认今天
        keyword: 可选关键词过滤（如 'AI'、'芯片'、'茅台'），空则返回所有
        max_events: 最大返回条数，默认 20
    """
    from financial_rag.rss_fetcher import search_news, fetch_all_news

    target_date = date or datetime.now().strftime("%Y-%m-%d")

    # 获取新闻 — 多源拉取以提高覆盖率
    all_items = []
    if keyword:
        result = search_news(keyword=keyword, max_news=max_events * 2)
        all_items.extend(result.get("items", []))
    else:
        result = fetch_all_news(max_news=max_events * 2)
        all_items.extend(result.get("items", []))

    # 按日期过滤
    filtered = []
    for item in all_items:
        pub_time = item.get("publish_time", "")
        # 尝试多种日期格式匹配
        if target_date in pub_time or pub_time.startswith(target_date):
            filtered.append({
                "title": item.get("title", ""),
                "content": item.get("content", "")[:300],
                "source": item.get("source", "unknown"),
                "publish_time": pub_time,
                "url": item.get("url", ""),
            })

    # 去重（按 title）
    seen = set()
    deduped = []
    for item in filtered:
        if item["title"] not in seen:
            seen.add(item["title"])
            deduped.append(item)

    # 截取
    deduped = deduped[:max_events]

    return {
        "date": target_date,
        "keyword": keyword,
        "total": len(deduped),
        "events": deduped,
    }


# ===================== 工具 2: 获取 K 线上下文 =====================

def fetch_kline_context(
    stock_code: str = "600519.SH",
    date: str = "",
    window_days: int = 10,
) -> Dict:
    """获取指定日期前后的 K 线数据，用于事件影响分析。

    拉取 date 前后各 window_days 个交易日的 K 线，
    计算事件前后的价格变化和波动。

    Args:
        stock_code: Tushare 股票代码，如 '600519.SH'（茅台）
        date: 事件日期 'YYYY-MM-DD'，默认最近交易日
        window_days: 前后各取几个交易日，默认 10
    """
    from financial_rag.tushare_client import (
        fetch_stock_kline, fetch_etf_kline, compute_kline_stats
    )

    target_date = date or datetime.now().strftime("%Y%m%d")
    # 统一格式为 YYYYMMDD
    clean_date = target_date.replace("-", "")

    # 多拉一些天以确保覆盖 window
    fetch_days = window_days * 2 + 10

    # 判断 ETF 还是股票
    is_etf = stock_code.startswith("51") or stock_code.startswith("159")
    try:
        if is_etf:
            df = fetch_etf_kline(stock_code, days=fetch_days)
        else:
            df = fetch_stock_kline(stock_code, days=fetch_days)
    except Exception as e:
        return {"error": f"K线获取失败: {e}", "stock_code": stock_code}

    if df.empty:
        return {"error": f"未获取到 {stock_code} 的K线数据", "stock_code": stock_code}

    # 按日期过滤到 window 范围
    if "date" in df.columns:
        df["date_str"] = df["date"].astype(str).str.replace("-", "")
        target_int = int(clean_date)

        # 找最接近目标日期的位置
        date_diffs = (df["date_str"].astype(int) - target_int).abs()
        center_idx = date_diffs.idxmin()

        # 取前后 window
        start_idx = max(0, center_idx - window_days)
        end_idx = min(len(df), center_idx + window_days + 1)
        df_window = df.iloc[start_idx:end_idx].copy()
    else:
        df_window = df.tail(window_days * 2)

    # 计算事件前后的关键指标
    kline_data = []
    for _, row in df_window.iterrows():
        kline_data.append({
            "date": str(row.get("date", ""))[:10],
            "open": float(row.get("open", 0)),
            "close": float(row.get("close", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "volume": int(row.get("volume", 0)),
            "pct_change": float(row.get("pct_chg", 0)) if "pct_chg" in row.index else 0.0,
        })

    # 计算事件前/后的平均涨跌幅
    center_date = target_date.replace("-", "")[:10]
    before = [k for k in kline_data if k["date"] < center_date]
    after = [k for k in kline_data if k["date"] >= center_date]

    before_avg_chg = (
        sum(k["pct_change"] for k in before) / len(before) if before else 0
    )
    after_avg_chg = (
        sum(k["pct_change"] for k in after) / len(after) if after else 0
    )

    return {
        "stock_code": stock_code,
        "date": target_date,
        "window_days": window_days,
        "data_points": len(kline_data),
        "kline": kline_data,
        "before_avg_change_pct": round(before_avg_chg, 2),
        "after_avg_change_pct": round(after_avg_chg, 2),
        "change_delta": round(after_avg_chg - before_avg_chg, 2),
    }


# ===================== 工具 3: LLM 事件影响评估 =====================

_llm_ref = {"llm": None}


def inject_event_llm(llm):
    """注入 LLM 实例给事件影响评估工具"""
    _llm_ref["llm"] = llm


def assess_event_impact(
    events: List[Dict],
    kline_context: Dict = None,
    stock_name: str = "",
) -> Dict:
    """使用 LLM 评估事件对市场的潜在影响。

    对每个事件分类为利好/利空/中性，并估算影响因子（0-10）。
    如果提供了 K 线数据，还会结合历史表现做判断。

    Args:
        events: 事件列表，每项包含 title, content, source
        kline_context: K 线上下文数据（可选，来自 fetch_kline_context）
        stock_name: 股票名称（如 '贵州茅台'），辅助 LLM 理解
    """
    if not events:
        return {"error": "无事件可评估", "assessments": []}

    llm = _llm_ref["llm"]
    if llm is None:
        return _fallback_assess(events)

    # 构建事件摘要
    event_text = ""
    for i, e in enumerate(events[:10], 1):
        title = e.get("title", "")
        content = e.get("content", "")[:150]
        event_text += f"{i}. [{e.get('source', '')}] {title}\n   {content}\n\n"

    # 构建 K 线摘要
    kline_text = ""
    if kline_context and "kline" in kline_context:
        kline_text = (
            f"\n## 近期K线数据\n"
            f"- 事件前平均涨跌: {kline_context.get('before_avg_change_pct', 'N/A')}%\n"
            f"- 事件后平均涨跌: {kline_context.get('after_avg_change_pct', 'N/A')}%\n"
            f"- 变化幅度: {kline_context.get('change_delta', 'N/A')}%\n"
            f"- 最近K线:\n"
        )
        for k in kline_context.get("kline", [])[-5:]:
            kline_text += (
                f"  {k['date']}: 开{k['open']} 收{k['close']} "
                f"涨跌{k['pct_change']}%\n"
            )

    prompt = f"""请分析以下事件对{'（' + stock_name + '）' if stock_name else 'A股市场'}的影响。

## 事件列表
{event_text}{kline_text}

请按以下 JSON 格式输出每个事件的影响评估:
{{
  "assessments": [
    {{
      "event": "事件标题（简短）",
      "impact": "bullish" 或 "bearish" 或 "neutral",
      "impact_label": "利好" 或 "利空" 或 "中性",
      "impact_factor": 0-10 的整数（影响程度，10=极大影响）,
      "reasoning": "简要理由（一句话）",
      "affected_sectors": ["受影响的板块/行业"]
    }}
  ],
  "overall_impact": "bullish" 或 "bearish" 或 "neutral",
  "overall_label": "综合利好/利空/中性",
  "overall_factor": 0-10,
  "summary": "一段总结性分析（100字以内）"
}}"""

    try:
        resp = llm.chat(messages=prompt, max_tokens=800, temperature=0.1)
        import json
        content = resp.content.strip()
        # 解析 JSON
        if "{" in content:
            parsed = json.loads(content[content.index("{"):content.rindex("}") + 1])
            return parsed
    except Exception as e:
        logger.warning(f"LLM 事件评估失败: {e}")

    return _fallback_assess(events)


def _fallback_assess(events: List[Dict]) -> Dict:
    """无 LLM 时的规则化评估"""
    bullish_keywords = [
        "增长", "上涨", "突破", "利好", "新高", "超预期", "获批", "盈利",
        "扩大", "景气", "回暖", "涨停", "买入", "增持", "推荐",
    ]
    bearish_keywords = [
        "下跌", "下降", "利空", "亏损", "违规", "处罚", "减持", "风险",
        "退市", "暴跌", "跌停", "下调", "预警", "亏损", "诉讼",
    ]

    assessments = []
    for e in events:
        text = e.get("title", "") + " " + e.get("content", "")[:200]
        bull_score = sum(1 for kw in bullish_keywords if kw in text)
        bear_score = sum(1 for kw in bearish_keywords if kw in text)

        if bull_score > bear_score:
            impact = "bullish"
            label = "利好"
            factor = min(10, 3 + bull_score)
        elif bear_score > bull_score:
            impact = "bearish"
            label = "利空"
            factor = min(10, 3 + bear_score)
        else:
            impact = "neutral"
            label = "中性"
            factor = 2

        assessments.append({
            "event": e.get("title", "")[:40],
            "impact": impact,
            "impact_label": label,
            "impact_factor": factor,
            "reasoning": "基于关键词规则判断",
            "affected_sectors": [],
        })

    # 综合
    total_factor = sum(a["impact_factor"] for a in assessments)
    bull_count = sum(1 for a in assessments if a["impact"] == "bullish")
    bear_count = sum(1 for a in assessments if a["impact"] == "bearish")

    if bull_count > bear_count:
        overall = "bullish"
        overall_label = "综合利好"
    elif bear_count > bull_count:
        overall = "bearish"
        overall_label = "综合利空"
    else:
        overall = "neutral"
        overall_label = "综合中性"

    return {
        "assessments": assessments,
        "overall_impact": overall,
        "overall_label": overall_label,
        "overall_factor": min(10, total_factor // max(len(assessments), 1) + 2),
        "summary": f"共 {len(assessments)} 个事件: {bull_count} 利好, {bear_count} 利空 (规则引擎)",
        "engine": "keyword_rules",
    }


# ===================== FunctionDef 定义 =====================

EVENT_IMPACT_TOOLS = [
    FunctionDef(
        name="fetch_date_events",
        description="获取指定日期发生的重要事件/新闻。当用户问'某天发生了什么'、"
                    "'2024年6月1日有什么大事'时使用。"
                    "从同花顺/新浪/东方财富获取当日新闻，按日期过滤。",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期 'YYYY-MM-DD'，默认今天", "default": ""},
                "keyword": {"type": "string", "description": "可选关键词过滤，如 'AI'、'芯片'", "default": ""},
                "max_events": {"type": "integer", "description": "最大返回条数", "default": 20},
            },
            "required": [],
        },
        callback=fetch_date_events,
        category="data",
        tags=["事件", "新闻", "日期", "历史"],
    ),
    FunctionDef(
        name="fetch_kline_context",
        description="获取指定日期前后的K线数据，用于分析事件对股价的影响。"
                    "当需要查看某事件前后的价格变化时使用。"
                    "数据源: Tushare Pro。",
        parameters={
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "Tushare代码，如 '600519.SH'", "default": "600519.SH"},
                "date": {"type": "string", "description": "事件日期 'YYYY-MM-DD'", "default": ""},
                "window_days": {"type": "integer", "description": "前后各取几天", "default": 10},
            },
            "required": [],
        },
        callback=fetch_kline_context,
        category="data",
        tags=["K线", "事件影响", "价格变化", "Tushare"],
    ),
    FunctionDef(
        name="assess_event_impact",
        description="评估事件对市场的潜在影响（利好/利空/中性 + 影响因子0-10）。"
                    "输入事件列表和可选的K线数据，返回每个事件的影响分类和综合判断。"
                    "通常配合 fetch_date_events 使用：先拉事件，再评估影响。",
        parameters={
            "type": "object",
            "properties": {
                "events": {"type": "array", "description": "事件列表，每项含 title/content/source",
                           "items": {"type": "object"}},
                "kline_context": {"type": "object", "description": "K线上下文（可选，来自 fetch_kline_context）", "default": None},
                "stock_name": {"type": "string", "description": "股票名称，辅助LLM理解", "default": ""},
            },
            "required": ["events"],
        },
        callback=assess_event_impact,
        category="analysis",
        tags=["事件影响", "利好利空", "影响因子", "LLM"],
    ),
]
