"""
L2 输出层质量指标 — 纯函数、离线确定、零 LLM 依赖。

对称 metrics.py（L1 检索层 IR 指标），但针对「生成输出质量」，聚焦 NEWS_BRIEF：
- numerical_fidelity: 复用 Guard l2_numerical_fidelity —— 检出编造数字
- source_grounding:   复用 Guard l1_source_grounding —— 检出无源声明
- field_completeness: NEWS_BRIEF 7 字段有效填充率 —— 新定制（Guard L4 的
                      rag/analysis 两套段落期望都不匹配简报的字段式结构）
- output_quality:     加权聚合成单一综合分

设计原则（对应 plan 决策 2）：复用 guard/rule_layers.py 的模块级纯函数，
不改 Guard 生产逻辑、不碰 reflector.py。所有函数不碰网络/LLM/随机，同输入必同输出。

输入约定:
- brief_output: NEWS_BRIEF 模板渲染后的输出文本
- source_docs:  检索源文档列表 [{"text": ...}, ...]（Guard l1/l2 的 sources 形状）
"""
import re
from typing import Dict, List

from financial_rag.guard.rule_layers import (
    l1_source_grounding,
    l2_numerical_fidelity,
)

# 综合分权重：数值忠实度最重 —— 新闻简报最怕编造数字/日期/涨跌幅
OUTPUT_QUALITY_WEIGHTS = {
    "numerical_fidelity": 0.4,
    "source_grounding": 0.3,
    "field_completeness": 0.3,
}

# NEWS_BRIEF 渲染字段 → 提取正则（严格对应 templates.NEWS_BRIEF_RENDER）
#   ## 📰 {event_title}
#   **时间**: {event_date} | **影响**: {impact_level}
#   **涉及**: {key_players} / **行业**: {sector_impact}
#   **市场**: {market_reaction} / **展望**: {outlook}
# event_title 走 _TITLE_PATTERN 单独处理，其余 6 个字段走下表。
_FIELD_PATTERNS = [
    ("event_date", r"\*\*时间\*\*:\s*([^|\n]+)"),
    ("impact_level", r"\*\*影响\*\*:\s*([^\n]+)"),
    ("key_players", r"\*\*涉及\*\*:\s*([^\n]+)"),
    ("sector_impact", r"\*\*行业\*\*:\s*([^\n]+)"),
    ("market_reaction", r"\*\*市场\*\*:\s*([^\n]+)"),
    ("outlook", r"\*\*展望\*\*:\s*([^\n]+)"),
]
_TITLE_PATTERN = re.compile(r"^##\s*📰\s*(.+?)\s*$", re.MULTILINE)
# SlotFiller.render 对未填充槽位写 "（{label}暂缺）"，判为未填充
_PLACEHOLDER_PATTERN = re.compile(r"（.*暂缺）")
_TOTAL_FIELDS = 7  # event_title + 上述 6 字段


def _is_filled(value: str) -> bool:
    """字段值是否为有效填充：非空、非「（xx暂缺）」占位符。"""
    v = (value or "").strip()
    if not v:
        return False
    if _PLACEHOLDER_PATTERN.search(v):
        return False
    return True


def field_completeness(brief_output: str) -> Dict:
    """NEWS_BRIEF 7 字段有效填充率。

    返回 {score, filled, total, missing_fields}；score = filled / 7 ∈ [0,1]。
    空值与「（xx暂缺）」占位符均判为未填充。
    """
    text = brief_output or ""
    missing: List[str] = []
    filled = 0

    title_m = _TITLE_PATTERN.search(text)
    if title_m and _is_filled(title_m.group(1)):
        filled += 1
    else:
        missing.append("event_title")

    for key, pat in _FIELD_PATTERNS:
        fm = re.search(pat, text)
        if fm and _is_filled(fm.group(1)):
            filled += 1
        else:
            missing.append(key)

    return {
        "score": filled / _TOTAL_FIELDS,
        "filled": filled,
        "total": _TOTAL_FIELDS,
        "missing_fields": missing,
    }


def evaluate_case(brief_output: str, source_docs: List[Dict]) -> Dict:
    """对单条输出样本跑全套质量指标（离线确定）。

    返回:
        numerical_fidelity / source_grounding / field_completeness / output_quality: [0,1]
        unmatched_numbers: 答案中无法在 source 溯源的「数字+单位」（编造嫌疑）
        unanchored:        无法锚定到任何 source 的句子
        missing_fields:    未有效填充的 NEWS_BRIEF 字段 key
    """
    l2 = l2_numerical_fidelity(brief_output, source_docs)
    l1 = l1_source_grounding(brief_output, source_docs)
    fc = field_completeness(brief_output)

    nf = l2["score"]
    sg = l1["score"]
    fcs = fc["score"]
    quality = (
        OUTPUT_QUALITY_WEIGHTS["numerical_fidelity"] * nf
        + OUTPUT_QUALITY_WEIGHTS["source_grounding"] * sg
        + OUTPUT_QUALITY_WEIGHTS["field_completeness"] * fcs
    )
    return {
        "numerical_fidelity": nf,
        "source_grounding": sg,
        "field_completeness": fcs,
        "output_quality": quality,
        "unmatched_numbers": l2.get("unmatched", []),
        "unanchored": l1.get("unanchored", []),
        "missing_fields": fc["missing_fields"],
    }
