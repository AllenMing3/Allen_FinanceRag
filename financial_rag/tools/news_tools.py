"""
新闻工具 — 可注册到 FunctionRegistry 的新闻能力

提供:
- fetch_news_report: 搜索财经新闻、过滤、保存为 Markdown 报告（纯数据操作）
- run_news_pipeline: 端到端新闻流水线（关键词 LLM 提取 + 拉取 + AI 摘要 + 拼接 Markdown）
  供 CLI 命令直接调起，coordinator 只需路由。
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)


def fetch_news_report(
    keyword: str,
    max_news: int = 30,
    output_dir: str = "",
    filename: str = "",
) -> Dict:
    """搜索财经新闻并保存为 Markdown 格式化报告。

    从东方财富全球财经快讯中按关键词搜索，过滤匹配项，
    保存为结构化的 .md 文件，返回摘要信息。

    Args:
        keyword: 搜索关键词，如 '央行降准'、'AI人工智能'、'茅台'
        max_news: 最大返回条数，默认 30
        output_dir: 输出目录，默认系统 output 目录
        filename: 自定义文件名（不含扩展名），默认自动生成
    """
    from financial_rag.config import config
    from financial_rag.news_fetcher import fetch_financial_news

    # 确定输出目录
    out_dir = output_dir or config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 获取新闻
    raw = fetch_financial_news(keyword=keyword, max_news=max_news * 2)
    items = raw.get("items", [])

    # 按关键词过滤（二次过滤，akshare 有时返回宽泛结果）
    filtered = []
    for item in items:
        title = item.get("title", "")
        content = item.get("content", "")
        text = title + " " + content
        # 分词式匹配：关键词拆分为多个子词，至少匹配一个
        kw_parts = [kw.strip() for kw in keyword.replace("、", ",").replace("，", ",").split(",") if kw.strip()]
        if any(p in text for p in kw_parts if len(p) >= 2):
            filtered.append(item)
        elif not kw_parts:
            filtered.append(item)

    filtered = filtered[:max_news]

    # 生成文件名
    today = datetime.now().strftime("%Y-%m-%d")
    safe_kw = keyword.replace("/", "_").replace("\\", "_").replace("、", "").replace("，", "")[:15]
    fname = filename or f"{today}_{safe_kw}_新闻汇总.md"
    filepath = os.path.join(out_dir, fname)

    # 构建 Markdown
    lines = [
        f"# {keyword} 新闻汇总", "",
        f"> 查询: {keyword}",
        f"> 日期: {today}",
        f"> 条数: {len(filtered)}",
        f"> 数据源: 东方财富全球财经快讯",
        "", "---", "",
    ]

    for i, item in enumerate(filtered, 1):
        title = item.get("title", "无标题")
        content = item.get("content", "")
        source = item.get("source", "未知")
        pub_time = item.get("publish_time", "")
        url = item.get("url", "")
        lines.extend([
            f"## {i}. {title}", "",
            f"- **来源**: {source}",
            f"- **时间**: {pub_time}",
            f"- **链接**: {url}", "",
            f">{content}", "",
            "---", "",
        ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "keyword": keyword,
        "total_found": len(filtered),
        "filepath": filepath,
        "headlines": [
            {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "publish_time": item.get("publish_time", ""),
                "sentiment": item.get("sentiment", ""),
            }
            for item in filtered[:20]
        ],
        "items": [
            {
                "title": item.get("title", ""),
                "content": item.get("content", "")[:500],
                "source": item.get("source", ""),
                "publish_time": item.get("publish_time", ""),
                "url": item.get("url", ""),
                "sentiment": item.get("sentiment", ""),
            }
            for item in filtered[:10]
        ],
    }


# ===================== 端到端流水线（吞掉 coordinator 里的业务逻辑） =====================

EN_TO_CN = {
    "ai": "人工智能", "AI": "人工智能", "robot": "机器人",
    "blockchain": "区块链", "meta": "元宇宙", "ev": "新能源",
}

TOPIC_MAP = {
    "AI": "人工智能", "ai": "人工智能", "芯片": "芯片", "半导体": "半导体",
    "5G": "5G", "通信": "通信", "云计算": "云计算", "大数据": "大数据",
    "新能源": "新能源", "智能汽车": "智能汽车", "智能驾驶": "智能驾驶",
    "机器人": "机器人", "智能制造": "智能制造",
}


def _extract_keywords(llm, query: str) -> List[str]:
    """用 LLM 从用户查询中提取搜索关键词（无 LLM 时退化）"""
    keywords = [query]
    if llm is None:
        return keywords

    try:
        resp = llm.chat(
            messages=f"用户问：{query}\n\n请提取用于搜索财经新闻的3-5个中文关键词，只输出逗号分隔的关键词，不要其他内容。",
            max_tokens=40,
        )
        keywords = [k.strip() for k in resp.content.replace("\n", "").replace("、", ",").split(",") if k.strip()]
    except Exception:
        pass

    # 英译中映射
    mapped = []
    for kw in keywords:
        mapped.append(kw)
        for en, cn in EN_TO_CN.items():
            if en.lower() in kw.lower() and cn not in mapped:
                mapped.append(cn)
    return list(dict.fromkeys(mapped))


def _generate_summary(llm, headlines: List[Dict], topic: str) -> str:
    """用 LLM 对新闻标题列表做摘要"""
    if llm is None or not headlines:
        return ""

    try:
        hl_text = "\n".join(
            f"- [{h.get('publish_time', '')[:10]}] {h['title']}"
            for h in headlines[:30]
        )
        resp = llm.chat(
            messages=f"以下是关于{topic}的最新财经新闻标题。请用300字以内做摘要，概括主要动态和趋势：\n\n{hl_text}",
            max_tokens=400,
        )
        return resp.content
    except Exception:
        return ""


def _append_summary_to_md(filepath: str, summary: str) -> None:
    """将 AI 摘要插入到已保存 Markdown 的第一个 --- 之后"""
    if not summary or not filepath or not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 1)
    if len(parts) == 2:
        new_content = parts[0] + "---\n\n" + summary + "\n\n## AI 摘要\n\n---" + parts[1]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


def run_news_pipeline(
    llm=None,
    *,
    query: str = "",
    output_dir: str = "",
    filename: str = "",
    summarize: bool = False,
    max_news: int = 30,
) -> Dict:
    """端到端新闻流水线：LLM 提取关键词 → 拉取新闻 → 保存 Markdown → 可选 AI 摘要。

    调用方（CLI coordinator）只需传参 + 读返回值 + 打印，
    所有业务逻辑内聚在此函数中。

    Args:
        llm: DashScope LLM 实例（可选，无则跳过关键词提取和摘要）
        query: 用户原始查询
        output_dir: 输出目录
        filename: 自定义文件名
        summarize: 是否生成 AI 摘要
        max_news: 最大新闻条数

    Returns:
        dict 包含 filepath, total_found, keywords, summary, 等
    """
    from financial_rag.config import config

    out_dir = output_dir or config.output_dir

    # 1. 关键词提取
    keywords = _extract_keywords(llm, query)
    main_kw = "、".join(keywords[:3])
    logger.info(f"新闻流水线: query='{query}' keywords='{main_kw}'")

    # 2. 拉取 + 保存
    data = fetch_news_report(
        keyword=main_kw, max_news=max_news,
        output_dir=out_dir, filename=filename,
    )

    # 3. 可选 AI 摘要
    summary = ""
    if summarize:
        summary = _generate_summary(llm, data.get("headlines", []), main_kw)
        _append_summary_to_md(data.get("filepath", ""), summary)

    return {
        **data,
        "keywords": keywords,
        "main_keyword": main_kw,
        "summary": summary,
        "has_summary": bool(summary),
    }


# ===================== FunctionDef 定义 =====================

NEWS_REPORT_TOOL = FunctionDef(
    name="fetch_news_report",
    description="搜索财经新闻并保存为 Markdown 格式化报告。"
                "当用户问'最近有什么XX新闻'、'搜索XX相关新闻'、'帮我找XX的新闻'时使用。"
                "返回文件路径和新闻列表，LLM 拿到后可进一步总结分析。",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "搜索关键词（中文），如 '央行降准'、'AI人工智能'、'茅台财报'",
            },
            "max_news": {
                "type": "integer",
                "description": "最大返回条数，默认 30",
                "default": 30,
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
    callback=fetch_news_report,
    category="data",
    tags=["新闻", "搜索", "报告", "快讯"],
)
