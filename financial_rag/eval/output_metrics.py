"""
L2 输出层质量指标 — 纯函数、离线确定、零 LLM 依赖。

对称 metrics.py（L1 检索层 IR 指标），但针对「生成输出质量」，多模板支持：
- numerical_fidelity: 复用 Guard l2_numerical_fidelity —— 检出编造数字
- source_grounding:   复用 Guard l1_source_grounding —— 检出无源声明
- field_completeness: 按模板字段规格计算有效填充率 —— 新定制（Guard L4 的
                      rag/analysis 两套段落期望都不匹配槽位模板的字段式结构）
- output_quality:     加权聚合成单一综合分

设计原则（对应 plan 决策 2）：复用 guard/rule_layers.py 的模块级纯函数，
不改 Guard 生产逻辑、不碰 reflector.py。所有函数不碰网络/LLM/随机，同输入必同输出。

输入约定:
- brief_output: 模板渲染后的输出文本（news_brief / quick_qa / deep_analysis）
- source_docs:  检索源文档列表 [{"text": ...}, ...]（Guard l1/l2 的 sources 形状）
- template:     模板名，默认 news_brief（向后兼容）；未知模板 ValueError
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

# 模板脚手架（标签词/标题前缀）：非声明内容，喂 Guard 前剥离。
# 否则 Guard L1 阈值收紧到 0.30 后，"**时间**: ... | **影响**: ..." 这类标签行
# 会因标签词与 source 零重叠被误判 unanchored（标签属模板，不属生成内容）。
_TEMPLATE_NOISE_RE = re.compile(
    r"\*\*(?:时间|影响|涉及|行业|市场|展望)\*\*\s*[:：]|^##\s*📰\s*|[|｜]", re.MULTILINE)

# ===================== 模板字段规格注册表（P0-1 多模板） =====================
# 每模板: fields = [(字段key, 带捕获组的提取正则)]，noise = 脚手架剥离正则。
# 字段提取严格对应 templates.py 各 *_RENDER 的渲染结构。

_DEEP_SECTIONS = [
    ("executive_summary", "摘要"), ("profitability", "盈利能力"),
    ("growth", "成长性"), ("health", "财务健康度"),
    ("valuation_hint", "估值参考"), ("recommendation", "投资建议"),
]

_TEMPLATE_SPECS = {
    "news_brief": {
        "fields": ([("event_title", _TITLE_PATTERN)]
                   + [(k, re.compile(p)) for k, p in _FIELD_PATTERNS]),
        "noise": _TEMPLATE_NOISE_RE,
    },
    "quick_qa": {
        "fields": [
            ("direct_answer", re.compile(r"###\s*回答\s*\n([\s\S]*?)(?=\n\*\*|\Z)")),
            ("supporting_data", re.compile(r"\*\*数据支撑\*\*:\s*([^\n]+)")),
            ("source_note", re.compile(r"\*\*来源\*\*:\s*([^\n]+)")),
            ("confidence_note", re.compile(r"\*\*可信度\*\*:\s*([^\n]+)")),
        ],
        "noise": re.compile(
            r"\*\*(?:数据支撑|来源|可信度)\*\*\s*[:：]|^###\s*回答\s*$", re.MULTILINE),
    },
    "deep_analysis": {
        "fields": [
            (key, re.compile(rf"###\s*{label}\s*\n([\s\S]*?)(?=\n###\s|\Z)"))
            for key, label in _DEEP_SECTIONS
        ],
        "noise": re.compile(
            r"^##\s*深度分析报告\s*$|^###\s*(?:摘要|盈利能力|成长性|财务健康度|估值参考|投资建议)\s*$",
            re.MULTILINE),
    },
}


def _get_template_spec(template: str) -> Dict:
    spec = _TEMPLATE_SPECS.get(template)
    if spec is None:
        raise ValueError(
            f"未知模板: {template!r}，可选: {sorted(_TEMPLATE_SPECS)}")
    return spec


def _is_filled(value: str) -> bool:
    """字段值是否为有效填充：非空、非「（xx暂缺）」占位符。"""
    v = (value or "").strip()
    if not v:
        return False
    if _PLACEHOLDER_PATTERN.search(v):
        return False
    return True


def field_completeness(brief_output: str, template: str = "news_brief") -> Dict:
    """模板字段有效填充率（news_brief 7 字段 / quick_qa 4 字段 / deep_analysis 6 段落）。

    返回 {score, filled, total, missing_fields}；score = filled / total ∈ [0,1]。
    空值与「（xx暂缺）」占位符均判为未填充。
    """
    spec = _get_template_spec(template)
    text = brief_output or ""
    missing: List[str] = []
    filled = 0

    for key, pat in spec["fields"]:
        fm = pat.search(text)
        if fm and _is_filled(fm.group(1)):
            filled += 1
        else:
            missing.append(key)

    total = len(spec["fields"])
    return {
        "score": filled / total,
        "filled": filled,
        "total": total,
        "missing_fields": missing,
    }


def _strip_template_scaffolding(brief_output: str, template: str = "news_brief") -> str:
    """剥离模板脚手架（字段标签/标题前缀/分隔符），仅留声明内容。"""
    spec = _get_template_spec(template)
    return spec["noise"].sub(" ", brief_output or "")


def evaluate_case(brief_output: str, source_docs: List[Dict],
                  template: str = "news_brief") -> Dict:
    """对单条输出样本跑全套质量指标（离线确定）。

    返回:
        numerical_fidelity / source_grounding / field_completeness / output_quality: [0,1]
        unmatched_numbers: 答案中无法在 source 溯源的「数字+单位」（编造嫌疑）
        unanchored:        无法锚定到任何 source 的句子
        missing_fields:    未有效填充的模板字段 key
    """
    l2 = l2_numerical_fidelity(_strip_template_scaffolding(brief_output, template), source_docs)
    l1 = l1_source_grounding(_strip_template_scaffolding(brief_output, template), source_docs)
    fc = field_completeness(brief_output, template)

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
