"""
context_builder.py — 检索结果 → 喂给 AI 的 grounding 统一组装层

对治两个系统性问题（见 docs/OPTIMIZATION_ROADMAP.md 方向三）:

1. 一致性差: 此前三条通路各截各的（深度分析 [:150]/新闻综合 [:800]/KB问答 不截），
   同一份 chunk 走不同通路信息量差 5 倍以上。本模块提供**唯一组装算法**，
   各通路只在 budget 数值上不同，且组装出的 grounding **同一份**同时供
   LLM 生成与 HallucinationGuard 打分，消除"生成用一套来源、打分用另一套"。

2. 黑盒: 此前 grounding 被腰斩到保真率 13.7% 却无人察觉（靠手写探针才测出）。
   本模块每次组装产出 ContextAssemblyTrace，把保真率、逐项取舍+原因、预算利用率
   变成**头等可见指标**，做完当场可查，而非全量测试后才发现。

设计原则: 纯函数式、无外部依赖、无副作用；不触碰检索/存储，只做"已召回 → 组装"。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 句子边界：中英文句末标点 + 换行
_SENT_END = re.compile(r'(?<=[。！？；!?;\n])')

# 各通路预算预设（chars）——同一算法，只差预算，杜绝 ad-hoc 截断
BUDGET_PRESETS: Dict[str, int] = {
    "deep_analysis": 3000,   # 深度分析（原 [:150]*3 = 450，腰斩重灾区）
    "quick_qa": 4000,        # KB 问答
    "synthesis": 6000,       # 新闻综合解读
}


def estimate_tokens(text: str) -> int:
    """粗估 token 数：CJK 每字≈1 token，其余按 ~4 字符/token。与 slot_filler 同口径。"""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - cjk
    return cjk + (other + 3) // 4


@dataclass
class ItemTrace:
    """单条检索项的组装取舍记录（可观测最小单元）。"""
    index: int
    source: str
    score: float
    available_chars: int
    kept_chars: int
    status: str            # "full" | "partial" | "dropped"
    reason: str = ""


@dataclass
class ContextAssemblyTrace:
    """一次组装的完整诊断快照——对标 HallucinationGuard 的 scorecard。"""
    budget_chars: int
    max_items: int
    min_score: float
    input_items: int            # 原始召回条数
    passed_filter: int          # 过 min_score 后条数
    available_chars: int        # 过滤后总可用字符（保真率分母）
    included_items: int         # 实际进入上下文的条数
    kept_chars: int             # 实际喂入的字符（保真率分子）
    dropped_items: int
    fidelity: float             # kept_chars / available_chars
    budget_utilization: float   # kept_chars / budget_chars
    estimated_tokens: int
    items: List[ItemTrace] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def format_report(self) -> str:
        lines = [
            "── Context Assembly Trace ──────────────────",
            f"  预算: {self.kept_chars}/{self.budget_chars} chars "
            f"(利用率 {self.budget_utilization:.0%})  ~{self.estimated_tokens} tokens",
            f"  召回: {self.input_items} 项 → 过滤(min_score={self.min_score}) {self.passed_filter} 项 "
            f"→ 纳入 {self.included_items} 项",
            f"  保真率: {self.fidelity:.1%}  ({self.kept_chars}/{self.available_chars} chars)",
        ]
        for it in self.items:
            lines.append(
                f"    [{it.index}] {it.status:<7} {it.source[:20]:<20} "
                f"score={it.score:.3f}  kept={it.kept_chars}/{it.available_chars}"
                + (f"  ← {it.reason}" if it.reason else "")
            )
        if self.warnings:
            lines.append("  警告:")
            for w in self.warnings:
                lines.append(f"    ⚠️ {w}")
        lines.append("────────────────────────────────────────────")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON 可序列化形式——供 API 透出前端 / 结构化日志记录。"""
        return {
            "budget_chars": self.budget_chars,
            "max_items": self.max_items,
            "min_score": self.min_score,
            "input_items": self.input_items,
            "passed_filter": self.passed_filter,
            "available_chars": self.available_chars,
            "included_items": self.included_items,
            "kept_chars": self.kept_chars,
            "dropped_items": self.dropped_items,
            "fidelity": round(self.fidelity, 4),
            "budget_utilization": round(self.budget_utilization, 4),
            "estimated_tokens": self.estimated_tokens,
            "warnings": self.warnings,
            "items": [
                {
                    "index": it.index, "source": it.source, "score": it.score,
                    "available_chars": it.available_chars, "kept_chars": it.kept_chars,
                    "status": it.status, "reason": it.reason,
                }
                for it in self.items
            ],
        }


@dataclass
class AssembledContext:
    """组装产物：喂 LLM 的文本 + 喂 Guard 的来源（二者同源）+ 诊断 trace。"""
    grounding_text: str
    source_items: List[Dict[str, Any]]   # [{text(kept), source, score}] — Guard 与 LLM 共用
    trace: ContextAssemblyTrace


def _normalize(item: Dict[str, Any]) -> Tuple[str, float, str]:
    """兼容两种召回结构：{text,score,source} 与 {text,score,meta:{source}}。"""
    text = item.get("text", "") or ""
    score = float(item.get("score", 0) or 0)
    source = item.get("source") or item.get("meta", {}).get("source", "") or ""
    return text, score, source


def _truncate_at_sentence(text: str, budget: int) -> Tuple[str, bool]:
    """在 budget 内按句子边界截断。返回 (截断文本, 是否命中干净句边界)。

    绝不切在半词：取 ≤budget 的最后一个句末标点为切点；若首句就超 budget 则
    硬切并返回 False（标记给 trace，暴露"预算过小"）。
    """
    if len(text) <= budget:
        return text, True
    window = text[:budget]
    ends = [m.end() for m in _SENT_END.finditer(window)]
    if ends:
        return window[:ends[-1]], True
    return window, False


class ContextBuilder:
    """检索结果 → grounding 的唯一组装器。各通路共用，仅 budget/max_items 不同。"""

    def __init__(
        self,
        budget_chars: int = 3000,
        max_items: int = 5,
        min_score: float = 0.0,
        respect_sentence: bool = True,
    ):
        self.budget_chars = budget_chars
        self.max_items = max_items
        self.min_score = min_score
        self.respect_sentence = respect_sentence

    @classmethod
    def for_path(cls, path: str, **overrides) -> "ContextBuilder":
        """按通路名取预算预设（deep_analysis/quick_qa/synthesis），保证算法一致。"""
        budget = BUDGET_PRESETS.get(path, 3000)
        return cls(budget_chars=budget, **overrides)

    def build(self, items: List[Dict[str, Any]]) -> AssembledContext:
        # 1) 归一化 + 过滤 + 按分排序（防御性，检索通常已排序）
        norm = [_normalize(it) for it in (items or [])]
        passed = [(t, s, src) for (t, s, src) in norm if s >= self.min_score]
        passed.sort(key=lambda x: x[1], reverse=True)
        available_chars = sum(len(t) for (t, _, _) in passed)

        # 2) 逐项组装，直到预算耗尽或达 max_items
        kept_total = 0
        blocks: List[str] = []
        source_items: List[Dict[str, Any]] = []
        item_traces: List[ItemTrace] = []
        included = dropped = 0

        for idx, (text, score, src) in enumerate(passed[:self.max_items]):
            remaining = self.budget_chars - kept_total
            avail = len(text)
            if remaining <= 0:
                item_traces.append(ItemTrace(idx, src, score, avail, 0, "dropped", "预算耗尽"))
                dropped += 1
                continue
            if avail <= remaining:
                kept, _ = text, True
                status = "full"
                reason = ""
            else:
                if self.respect_sentence:
                    kept, clean = _truncate_at_sentence(text, remaining)
                else:
                    kept, clean = text[:remaining], False
                # 预算不足以容纳任何实质内容（截断后仅剩空白）→ 判为丢弃，
                # 不把 "\n" 之类的垃圾片段喂给 LLM/Guard
                if not kept.strip():
                    item_traces.append(
                        ItemTrace(idx, src, score, avail, 0, "dropped", "预算不足(仅剩空白)")
                    )
                    dropped += 1
                    continue
                status = "partial"
                reason = "句边界截断" if clean else "硬切(首句超预算)"
            kept_chars = len(kept)
            kept_total += kept_chars
            included += 1
            blocks.append(f"[{idx}] {src} (相关度 {score:.2f})\n{kept}")
            source_items.append({"text": kept, "source": src, "score": round(score, 4)})
            item_traces.append(ItemTrace(idx, src, score, avail, kept_chars, status, reason))

        # 过滤/上限导致的丢弃也纳入 trace（预算外被裁的项）
        for idx, (text, score, src) in enumerate(passed[self.max_items:]):
            item_traces.append(
                ItemTrace(self.max_items + idx, src, score, len(text), 0, "dropped", "超出 max_items")
            )
            dropped += 1

        grounding_text = "\n\n".join(blocks)
        fidelity = (kept_total / available_chars) if available_chars > 0 else 0.0

        trace = ContextAssemblyTrace(
            budget_chars=self.budget_chars,
            max_items=self.max_items,
            min_score=self.min_score,
            input_items=len(norm),
            passed_filter=len(passed),
            available_chars=available_chars,
            included_items=included,
            kept_chars=kept_total,
            dropped_items=dropped,
            fidelity=fidelity,
            budget_utilization=(kept_total / self.budget_chars) if self.budget_chars else 0.0,
            estimated_tokens=estimate_tokens(grounding_text),
            items=item_traces,
        )
        self._warn(trace)
        return AssembledContext(grounding_text=grounding_text, source_items=source_items, trace=trace)

    @staticmethod
    def _warn(trace: ContextAssemblyTrace) -> None:
        """把'静默腰斩'变成显式警告——这是可观测性的关键，杜绝黑盒。"""
        if trace.included_items == 0:
            trace.warnings.append("无任何内容进入上下文（召回为空或全部被过滤/裁掉）")
        elif trace.fidelity < 0.5:
            trace.warnings.append(
                f"保真率过低({trace.fidelity:.0%})：大量召回内容被预算裁掉，"
                f"AI 只看到片段——考虑调高 budget 或减少 max_items"
            )
        hard_cut = [it for it in trace.items if it.reason.startswith("硬切")]
        if hard_cut:
            trace.warnings.append(f"{len(hard_cut)} 项因首句超预算被硬切（可能断在半句）")
