"""
Test ContextBuilder — 检索结果→grounding 统一组装层。

本文件的存在意义：用**可复跑的测试**证明 context-builder 改造的实际效果，
杜绝"改完黑盒、全量测试才发现"。核心是 TestGroundingNotGutted——
它锁定"深度分析 grounding 不再被 [:150]*3 腰斩"，退回旧截断即变红。

无需 API key / 真实 KB：全部用假检索器 + conftest 样本文本。
"""
import json

import pytest

from financial_rag.context_builder import ContextBuilder
from tests.conftest import (
    SAMPLE_AI_FINANCIAL_REPORT,
    SAMPLE_AI_NEWS,
    SAMPLE_AI_FUNDING,
)


def _fake_items():
    """模拟检索召回：5 条真实金融文本，分数递减（均 >150 字、均 ≥0.4 阈值）。"""
    texts = [
        SAMPLE_AI_FINANCIAL_REPORT,
        SAMPLE_AI_NEWS,
        SAMPLE_AI_FUNDING,
        SAMPLE_AI_FINANCIAL_REPORT,
        SAMPLE_AI_NEWS,
    ]
    scores = [0.92, 0.81, 0.68, 0.55, 0.43]
    return [
        {"text": t, "score": s, "meta": {"source": f"研报_{i+1}"}}
        for i, (t, s) in enumerate(zip(texts, scores))
    ]


# ===========================================================================
# 核心回归：证明"效果"——grounding 不再被腰斩
# ===========================================================================
class TestGroundingNotGutted:
    """本次改造要'证明效果'的关键测试。若退回旧的 [:150]*3 固定截断，这里会红。"""

    def test_new_grounding_beats_old_150_truncation(self):
        items = _fake_items()
        # 旧行为（analysis.py 原 L388）：kb_sources[:3] 各 text[:150]
        old_kept = sum(min(len(it["text"]), 150) for it in items[:3])
        old_items = min(3, len(items))
        # 新行为：ContextBuilder 统一组装（deep_analysis 预算 3000）
        ctx = ContextBuilder.for_path("deep_analysis").build(items)

        # ① 纳入条数：旧只取 3 条，新在预算内全取
        assert ctx.trace.included_items > old_items, (
            f"纳入条数应增加：旧={old_items} → 新={ctx.trace.included_items}"
        )
        # ② 喂入信息量：新显著多于旧的 450 字上限
        assert ctx.trace.kept_chars > old_kept, (
            f"grounding 喂入量应提升：旧[:150]*3={old_kept}字 → 新={ctx.trace.kept_chars}字"
        )
        # ③ 预算充足时零腰斩：保真率满格
        assert ctx.trace.fidelity == pytest.approx(1.0), (
            f"预算充足时保真率应=100%，实测 {ctx.trace.fidelity:.1%}"
        )


# ===========================================================================
# 可观测性：组装过程全程可见（对治黑盒）
# ===========================================================================
class TestObservability:
    def test_trace_exposes_fidelity_and_per_item_status(self):
        ctx = ContextBuilder.for_path("deep_analysis").build(_fake_items())
        trace = ctx.trace
        assert 0.0 <= trace.fidelity <= 1.0
        assert trace.included_items == len(trace.items) or trace.dropped_items > 0
        # 每一项都有可诊断的状态与来源
        for it in trace.items:
            assert it.status in ("full", "partial", "dropped")
            assert it.source

    def test_budget_exhaustion_drops_items_with_reason(self):
        # 预算极小 → 必有项被丢弃，且给出明确原因（不静默）
        ctx = ContextBuilder(budget_chars=100, max_items=5).build(_fake_items())
        dropped = [it for it in ctx.trace.items if it.status == "dropped"]
        assert dropped, "预算过小应有项被丢弃"
        assert all(it.reason for it in dropped), "每个丢弃项都必须带原因"

    def test_no_whitespace_only_fragment_fed(self):
        # 回归锁定：样本文本均以 \n 开头，预算耗尽时曾命中开头 \n 产生 1 字垃圾片段。
        # grounding 绝不得包含纯空白内容（否则喂给 LLM/Guard 的是噪音）。
        ctx = ContextBuilder(budget_chars=100, max_items=5).build(_fake_items())
        assert ctx.source_items, "应至少纳入首项"
        for si in ctx.source_items:
            assert si["text"].strip(), f"grounding 含纯空白片段: {si['text']!r}"

    def test_low_fidelity_emits_warning(self):
        # 腰斩必须被显式警告暴露——这是'不再黑盒'的底线
        ctx = ContextBuilder(budget_chars=100, max_items=5).build(_fake_items())
        assert ctx.trace.fidelity < 0.5
        assert any("保真率过低" in w for w in ctx.trace.warnings)

    def test_trace_is_json_serializable(self):
        ctx = ContextBuilder.for_path("deep_analysis").build(_fake_items())
        d = ctx.trace.to_dict()
        json.dumps(d, ensure_ascii=False)  # 不抛异常 = 可透出前端/日志
        for key in ("fidelity", "kept_chars", "available_chars", "items", "warnings"):
            assert key in d


# ===========================================================================
# 一致性：喂 LLM 的文本 与 喂 Guard 的来源同源
# ===========================================================================
class TestConsistency:
    def test_grounding_text_and_guard_sources_same_source(self):
        ctx = ContextBuilder.for_path("deep_analysis").build(_fake_items())
        # grounding_text（喂 LLM）应完整包含每个 source_item（喂 Guard）的 kept 文本
        for si in ctx.source_items:
            assert si["text"] in ctx.grounding_text
        # 观测数字与实际组装一致（trace 不虚报）
        assert sum(len(si["text"]) for si in ctx.source_items) == ctx.trace.kept_chars

    def test_partial_truncation_lands_on_sentence_boundary(self):
        text = "公司营收增长36%。净亏损收窄28%。研发投入占比37%。算力利用率达85%。"
        items = [{"text": text, "score": 0.9, "meta": {"source": "研报"}}]
        ctx = ContextBuilder(budget_chars=20, max_items=5).build(items)
        assert ctx.trace.items[0].status == "partial"
        kept = ctx.source_items[0]["text"]
        # 关键：句边界截断，绝不切在半句/半词
        assert kept.endswith(("。", "！", "？", "；", "\n")), f"应落在句边界，实测结尾 {kept!r}"
        assert kept != text and len(kept) <= 20


# ===========================================================================
# 端到端接线：analyze_news_text 真的用上了 ContextBuilder
# ===========================================================================
class TestWiringIntoDeepAnalysis:
    class _FakeRetriever:
        def __init__(self, items):
            self._items = items

        def search_with_scores(self, query, top_k=5):
            return self._items[:top_k], None

    def test_analyze_news_returns_grounding_and_trace(self):
        from financial_rag.services.analysis import analyze_news_text

        items = _fake_items()
        result = analyze_news_text(
            SAMPLE_AI_NEWS,
            llm=None,  # 走规则回退，无需 API key
            retriever=self._FakeRetriever(items),
            kb_built=True,
        )
        # 新增的可观测字段必须在
        assert "kb_grounding" in result
        trace = result["kb_search_info"]["context_trace"]
        assert "fidelity" in trace
        assert trace["fidelity"] > 0.9, f"深度分析 grounding 保真率过低: {trace['fidelity']:.1%}"
        # kb_grounding 条数与 trace 记录的纳入数一致
        assert len(result["kb_grounding"]) == trace["included_items"]
