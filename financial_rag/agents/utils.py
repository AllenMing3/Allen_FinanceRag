"""
Agents 共享工具函数
"""
from typing import List, Dict


def build_news_context(items: List[Dict], max_items: int = 10) -> str:
    """将新闻元数据列表格式化为 LLM 可理解的上下文文本。

    Args:
        items: 新闻元数据列表，每项含 title/keyword/publish_time
        max_items: 最多取前 N 条

    Returns:
        格式化后的多行字符串，空列表返回空字符串
    """
    if not items:
        return ""
    lines = []
    for m in items[:max_items]:
        parts = []
        if m.get("title"):
            parts.append(m["title"])
        if m.get("keyword"):
            parts.append(f"关键词: {m['keyword']}")
        if m.get("publish_time"):
            parts.append(f"时间: {m['publish_time']}")
        if parts:
            lines.append("- " + " | ".join(parts))
    return "\n".join(lines)
