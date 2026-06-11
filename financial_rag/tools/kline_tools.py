"""
K线工具 — 可注册到 FunctionRegistry 的 ETF/股票 K 线分析能力

提供:
- fetch_kline_report: 搜索股票/ETF、拉取 K 线、计算统计、保存为 Markdown 分析报告（纯数据操作）
- run_kline_pipeline: 端到端 K 线流水线（LLM 提取参数 + 拉取 + AI 技术分析 + 拼接 Markdown）
  供 CLI 命令直接调起，coordinator 只需路由。

数据源: Tushare Pro (https://tushare.pro)
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


def fetch_etf_kline_report(
    keyword: str,
    days: int = 30,
    etf_code: str = "",
    output_dir: str = "",
    filename: str = "",
) -> Dict:
    """获取 ETF/股票 K 线数据并生成统计分析报告。

    按关键词搜索相关 ETF/股票，拉取历史 K 线数据，
    计算基础技术统计指标，保存为 Markdown 分析文件。

    Args:
        keyword: 主题关键词，如 '人工智能'、'芯片'、'半导体'、'新能源'、'茅台'
        days: 回溯交易天数，默认 30
        etf_code: 指定代码（如 '510300.SH'），空则自动搜索第一条
        output_dir: 输出目录，默认系统 output 目录
        filename: 自定义文件名（不含扩展名），默认自动生成
    """
    from financial_rag.config import config
    from financial_rag.tushare_client import (
        search_etf, search_stock, fetch_etf_kline, fetch_stock_kline, compute_kline_stats
    )

    # 主题词映射
    topic_map = {
        "AI": "人工智能", "ai": "人工智能",
        "芯片": "芯片", "半导体": "半导体",
        "5G": "5G", "通信": "通信",
        "云计算": "云计算", "大数据": "大数据",
        "新能源": "新能源", "智能汽车": "智能汽车",
        "智能驾驶": "智能驾驶", "机器人": "机器人",
        "智能制造": "智能制造",
    }
    keyword = topic_map.get(keyword, keyword)

    # 确定输出目录
    out_dir = output_dir or config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 搜索 ETF 或股票
    results = search_etf(keyword, limit=10)
    is_etf = True
    if not results:
        results = search_stock(keyword, limit=10)
        is_etf = False
    if not results:
        return {"error": f"未找到与 '{keyword}' 相关的 ETF/股票", "keyword": keyword, "results": []}

    # 选定目标
    target = results[0]
    if etf_code:
        matched = [e for e in results if e.get("ts_code", "") == etf_code or etf_code in e.get("ts_code", "")]
        if matched:
            target = matched[0]

    ts_code = target.get("ts_code", "")
    name = target.get("name", keyword)

    # 获取 K 线
    if is_etf:
        df = fetch_etf_kline(ts_code, days=days)
    else:
        df = fetch_stock_kline(ts_code, days=days)
    if df.empty:
        return {
            "error": f"未获取到 {ts_code} 的 K 线数据",
            "ts_code": ts_code,
            "name": name,
        }

    # 计算统计
    stats = compute_kline_stats(df)

    # 生成 Markdown
    today = datetime.now().strftime("%Y-%m-%d")
    safe_name = name.replace("/", "_").replace("\\", "_")
    fname = filename or f"{today}_{safe_name}_K线分析.md"
    filepath = os.path.join(out_dir, fname)

    col_map = {
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量", "amount": "成交额",
    }
    table_cols = [c for c in col_map if c in df.columns]
    header = "| " + " | ".join(col_map[c] for c in table_cols) + " |"
    sep = "| " + " | ".join("---" for _ in table_cols) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in table_cols:
            v = row[c]
            if c == "date":
                vals.append(str(v)[:10])
            elif c in ("volume",):
                vals.append(f"{v/10000:.0f}万")
            elif c in ("amount",):
                vals.append(f"{v/10000:.0f}万")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    table_md = "\n".join([header, sep] + rows)

    lines = [
        f"# {name} ({ts_code}) K线分析", "",
        f"> 查询: {keyword}",
        f"> 日期: {today}",
        f"> 回溯: {days} 个交易日",
        f"> 数据源: Tushare Pro",
        "", "---", "",
        "## 基础统计", "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 最新收盘价 | {stats.get('latest_close', '-')} |",
        f"| 区间最高 | {stats.get('period_high', '-')} |",
        f"| 区间最低 | {stats.get('period_low', '-')} |",
        f"| 区间涨跌幅 | {stats.get('period_change_pct', '-')}% |",
        f"| 上涨天数 | {stats.get('up_days', '-')} |",
        f"| 下跌天数 | {stats.get('down_days', '-')} |",
    ]
    if stats.get("avg_volume"):
        lines.append(f"| 平均成交量 | {stats.get('avg_volume', 0)/10000:.0f}万 |")
    if stats.get("ma5"):
        lines.append(f"| MA5 | {stats['ma5']} |")
    if stats.get("ma10"):
        lines.append(f"| MA10 | {stats['ma10']} |")

    lines += ["", "## K线数据", "", table_md]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 构建 K 线摘要（JSON 友好，不塞完整 DataFrame）
    kline_summary = []
    for _, row in df.tail(10).iterrows():
        kline_summary.append({
            "date": str(row.get("date", ""))[:10],
            "open": float(row.get("open", 0)),
            "close": float(row.get("close", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "volume": int(row.get("volume", 0)),
        })

    return {
        "keyword": keyword,
        "ts_code": ts_code,
        "name": name,
        "lookback_days": days,
        "data_points": len(df),
        "filepath": filepath,
        "stats": {k: v for k, v in stats.items() if v is not None},
        "kline_tail": kline_summary,
        "alternatives": [
            {"ts_code": e.get("ts_code", ""), "name": e.get("name", "")} for e in results[:5]
        ],
    }


# ===================== 端到端流水线（吞掉 coordinator 里的业务逻辑） =====================

def _extract_kline_params(llm, query: str, default_days: int = 30) -> tuple:
    """用 LLM 从用户查询中提取 keyword + 回溯天数"""
    keyword = query
    lookback_days = default_days
    if llm is None:
        return keyword, lookback_days

    try:
        import json as _json
        resp = llm.chat(
            messages=(
                f"从用户查询中提取两个信息，只输出 JSON：\n"
                f"1. keyword: ETF主题关键词（如 人工智能、芯片、半导体、新能源，只输出一个词）\n"
                f"2. days: 回溯天数数字（如 30、60、90，没有则默认30）\n\n"
                f"用户查询: {query}\n\n"
                f'输出格式: {{"keyword":"xx","days":30}}'
            ),
            max_tokens=60,
        )
        content = resp.content.strip()
        if "{" in content:
            parsed = _json.loads(content[content.index("{"):content.rindex("}") + 1])
            keyword = parsed.get("keyword", keyword)
            lookback_days = int(parsed.get("days", lookback_days))
    except Exception:
        pass

    # 主题词标准化
    topic_map = {
        "AI": "人工智能", "ai": "人工智能", "芯片": "芯片", "半导体": "半导体",
        "5G": "5G", "通信": "通信", "云计算": "云计算", "大数据": "大数据",
        "新能源": "新能源", "智能汽车": "智能汽车", "智能驾驶": "智能驾驶",
        "机器人": "机器人", "智能制造": "智能制造",
    }
    keyword = topic_map.get(keyword, keyword)
    return keyword, lookback_days


def _generate_analysis(llm, data: Dict) -> str:
    """用 LLM 对 K 线统计数据做技术分析"""
    if llm is None:
        return ""

    stats = data.get("stats", {})
    try:
        stats_str = (
            f"标的: {data.get('name')} ({data.get('ts_code')})\n"
            f"周期: 近{data.get('lookback_days')}个交易日\n"
            f"最新收盘: {stats.get('latest_close')}\n"
            f"区间涨跌幅: {stats.get('period_change_pct')}%\n"
            f"区间最高: {stats.get('period_high')}, 最低: {stats.get('period_low')}\n"
            f"MA5: {stats.get('ma5')}, MA10: {stats.get('ma10')}\n"
            f"上涨天数: {stats.get('up_days')}, 下跌天数: {stats.get('down_days')}"
        )
        resp = llm.chat(
            messages=(
                f"以下是该标的近期的 K 线数据统计，请用200字以内做简要技术分析，"
                f"包括趋势判断、支撑/压力位、短期展望：\n\n{stats_str}"
            ),
            max_tokens=300,
        )
        return resp.content
    except Exception:
        return ""


def _append_analysis_to_md(filepath: str, analysis: str) -> None:
    """将 AI 技术分析插入到已保存 Markdown 的 ## 基础统计 之前"""
    if not analysis or not filepath or not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    idx = content.find("## 基础统计")
    if idx >= 0:
        new_content = content[:idx] + analysis + "\n\n## AI 技术分析\n\n" + content[idx:]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


def run_kline_pipeline(
    llm=None,
    *,
    query: str = "",
    keyword: str = "",
    days: int = 30,
    etf_code: str = "",
    output_dir: str = "",
    filename: str = "",
    summarize: bool = False,
) -> Dict:
    """端到端 K 线流水线：LLM 提取参数 → 拉取 K 线 → 保存 Markdown → 可选 AI 技术分析。

    调用方（CLI coordinator）只需传参 + 读返回值 + 打印，
    所有业务逻辑内聚在此函数中。

    Args:
        llm: DashScope LLM 实例（可选，无则跳过参数提取和技术分析）
        query: 用户原始查询（用于 LLM 提取参数）
        keyword: ETF 主题关键词（空则从 query 提取）
        days: 回溯交易天数
        etf_code: 指定 ETF 代码
        output_dir: 输出目录
        filename: 自定义文件名
        summarize: 是否生成 AI 技术分析

    Returns:
        dict 包含 etf_code, etf_name, filepath, stats, analysis, 等
    """
    from financial_rag.config import config

    out_dir = output_dir or config.output_dir

    # 1. 参数提取（keyword 优先，否则从 query 用 LLM 提取）
    if not keyword and llm is not None:
        keyword, days = _extract_kline_params(llm, query, default_days=days)
    elif not keyword:
        keyword = query

    logger.info(f"K线流水线: keyword='{keyword}' days={days} etf_code='{etf_code}'")

    # 2. 拉取 + 保存
    data = fetch_etf_kline_report(
        keyword=keyword, days=days, etf_code=etf_code,
        output_dir=out_dir, filename=filename,
    )

    # 3. 可选 AI 技术分析
    analysis = ""
    if summarize and "error" not in data:
        analysis = _generate_analysis(llm, data)
        _append_analysis_to_md(data.get("filepath", ""), analysis)

    return {
        **data,
        "keyword_used": keyword,
        "analysis": analysis,
        "has_analysis": bool(analysis),
    }


# ===================== FunctionDef 定义 =====================

KLINE_REPORT_TOOL = FunctionDef(
    name="fetch_kline_report",
    description="获取股票/ETF K 线数据并生成统计分析报告。"
                "当用户问'XX 最近走势'、'看看半导体 ETF'、'分析茅台 K线'时使用。"
                "自动搜索最匹配的股票/ETF、拉取历史 K 线、计算技术指标、保存为 Markdown 文件。"
                "返回 K 线数据和统计指标，LLM 拿到后可进一步做技术分析解读。"
                "数据源: Tushare Pro。",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "主题关键词，如 '人工智能'、'芯片'、'茅台'、'新能源'、'5G'",
            },
            "days": {
                "type": "integer",
                "description": "回溯交易天数，默认 30",
                "default": 30,
            },
            "etf_code": {
                "type": "string",
                "description": "指定代码（如 '510300.SH'、'600519.SH'），空则自动搜索",
                "default": "",
            },
            "output_dir": {
                "type": "string",
                "description": "输出目录路径，默认系统 output 目录",
                "default": "",
            },
            "filename": {
                "type": "string",
                "description": "自定义文件名（不含扩展名），默认自动生成",
                "default": "",
            },
        },
        "required": ["keyword"],
    },
    callback=fetch_etf_kline_report,
    category="data",
    tags=["K线", "ETF", "股票", "技术分析", "行情", "Tushare"],
)
