"""
SlotFiller 引擎机制测试 — 槽位输出强化

离线确定性验证（无 API / 无网络 / 无 Phoenix）：
- 阶段机制：phase 顺序 + 前序结果注入（prev_context）
- 评分激励：诚实拒答不罚（0.6）/ 编造数字扣分 / 有据数字满分
- grounding：ContextBuilder 预算 + context_trace 可观测
- 清洗：多层废话前缀 / 引号包裹
- 渲染契约：（label暂缺）占位符（output_metrics 正则依赖）/ 裸 {key} 不泄漏
- 失败兜底：LLM 异常 → default + score 0.1 + error
- 统计：parallel_gain 范围 / 流式 TTFT
- prompt 分层：默认 system_prompt == prompts.SLOT_FILLER_SYSTEM

stub LLM 只测引擎机制（组装/清洗/评分/阶段/渲染），不评估生成质量；
真实质量需配 API key 实测。

运行: python -m pytest tests/test_slot_filler.py -v
"""
import re
import time
from types import SimpleNamespace

import pytest

from financial_rag import prompts
from financial_rag.slot_filler import SlotFiller, SlotResult, FillStats
from financial_rag.templates import SlotDef, SlottedTemplate


# ===================== 确定性 stub（符合 DashScopeLLM.chat 契约） =====================

class StubLLM:
    """按 prompt 关键词返回 canned 输出，记录调用顺序。"""

    def __init__(self, default="填充值", by_keyword=None, fail=False):
        self.default = default
        self.by_keyword = by_keyword or {}
        self.fail = fail
        self.calls = []

    def chat(self, messages, system=None, **kwargs):
        prompt = messages if isinstance(messages, str) else str(messages)
        self.calls.append(prompt)
        if self.fail:
            # ValueError 不触发 LLMCaller 的 RuntimeError 重试，快速失败
            raise ValueError("stub failure")
        for kw, resp in self.by_keyword.items():
            if kw in prompt:
                return SimpleNamespace(content=resp, usage={"total_tokens": len(resp)})
        return SimpleNamespace(content=self.default, usage={"total_tokens": len(self.default)})


class StreamStubLLM(StubLLM):
    """带 chat_stream 的 stub，用于 TTFT 测量路径。"""

    def chat_stream(self, messages, system=None, max_tokens=None):
        time.sleep(0.002)
        yield "填充", None
        yield "值", "stop"


def _tmpl(slots, phases, render):
    return SlottedTemplate(name="test", description="测试模板",
                           slots=slots, phases=phases, render=render)


# ===================== 阶段机制 =====================

class TestPhaseMechanics:
    def test_phase_order_and_prev_context(self):
        stub = StubLLM(by_keyword={"输出甲槽内容": "甲公司"})
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl(
            slots=[SlotDef("a", "甲槽", prompt="输出甲槽内容"),
                   SlotDef("b", "乙槽", prompt="输出乙槽内容")],
            phases=[["a"], ["b"]],
            render="{a} / {b}",
        )
        stats = filler.fill(tmpl, query="测试-阶段", context_docs=["背景资料"])
        assert stats.filled_slots == 2
        # phase2 的 prompt 必须含 phase1 的填充值（已知信息注入）
        b_calls = [c for c in stub.calls if "输出乙槽内容" in c]
        assert len(b_calls) == 1
        assert "甲公司" in b_calls[0]
        # 顺序：甲先于乙
        a_idx = next(i for i, c in enumerate(stub.calls) if "输出甲槽内容" in c)
        b_idx = next(i for i, c in enumerate(stub.calls) if "输出乙槽内容" in c)
        assert a_idx < b_idx

    def test_parallel_single_phase_fill(self):
        stub = StubLLM(default="有效值")
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl(
            slots=[SlotDef(f"s{i}", f"槽{i}", prompt=f"输出槽{i}内容") for i in range(3)],
            phases=[["s0", "s1", "s2"]],
            render="{s0} {s1} {s2}",
        )
        stats = filler.fill(tmpl, query="测试-并行", context_docs=[])
        assert stats.filled_slots == 3
        assert stats.total_slots == 3


# ===================== 评分激励 =====================

class TestScoringIncentives:
    def setup_method(self):
        self.filler = SlotFiller(llm=StubLLM())
        self.slot = SlotDef("x", "测试槽", prompt="p", max_tokens=80)

    def test_honest_refusal_fixed_score(self):
        # 诚实拒答是合规行为：固定 0.6，不与长度罚叠加
        assert self.filler._score_slot(self.slot, "信息不足") == 0.6
        assert self.filler._score_slot(self.slot, "根据已有信息无法判断") == 0.6

    def test_empty_value_zero(self):
        assert self.filler._score_slot(self.slot, "") == 0.0

    def test_honest_no_data_not_punished(self):
        # supporting_data 槽位指令明确要求"没有则输出'无'"，不应被长度罚惩罚
        assert self.filler._score_slot(self.slot, "无") == 0.6

    def test_fabricated_number_penalized(self):
        score = self.filler._score_slot(
            self.slot, "营收9000亿元", grounding="营收50.3亿元，同比增长36%")
        assert score <= 0.6

    def test_grounded_number_full_score(self):
        score = self.filler._score_slot(
            self.slot, "营收50.3亿元", grounding="营收50.3亿元，同比增长36%")
        assert score == 1.0


# ===================== grounding 纪律 =====================

class TestGroundingDiscipline:
    def test_budget_and_trace(self):
        stub = StubLLM()
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl([SlotDef("a", "甲", prompt="输出甲")], [["a"]], "{a}")
        long_doc = "测试内容句子。" * 1500  # ~10500 字，远超 quick_qa 4000 预算
        stats = filler.fill(tmpl, query="测试-预算", context_docs=[long_doc])
        # trace 可观测 + 预算生效
        assert "fidelity" in stats.context_trace
        assert stats.context_trace["kept_chars"] <= 4000
        assert stats.context_trace["fidelity"] < 1.0
        # 注入 LLM 的参考段受预算约束（不整段灌 10000 字）
        assert len(stub.calls[0]) < 5000

    def test_empty_docs_no_crash(self):
        stub = StubLLM()
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl([SlotDef("a", "甲", prompt="输出甲")], [["a"]], "{a}")
        stats = filler.fill(tmpl, query="测试-空上下文", context_docs=[])
        assert stats.filled_slots == 1
        assert isinstance(stats.context_trace, dict)


# ===================== 清洗 =====================

class TestCleaning:
    def setup_method(self):
        self.filler = SlotFiller(llm=StubLLM())
        self.slot = SlotDef("x", "测试槽", prompt="p")

    def test_multi_layer_prefix_strip(self):
        assert self.filler._clean_slot_value(
            "好的，根据参考信息，营收增长", self.slot) == "营收增长"

    def test_quote_strip(self):
        assert self.filler._clean_slot_value('"营收增长"', self.slot) == "营收增长"
        assert self.filler._clean_slot_value("“营收增长”", self.slot) == "营收增长"


# ===================== 渲染契约 =====================

class TestRenderContract:
    def test_unfilled_placeholder_format(self):
        # （label暂缺）格式被 output_metrics._PLACEHOLDER_PATTERN 依赖
        filler = SlotFiller(llm=StubLLM())
        tmpl = _tmpl([SlotDef("a", "直接回答", prompt="p")], [["a"]], "答：{a}")
        stats = FillStats(template_name="test", total_slots=1)
        stats.slot_results = {"a": SlotResult(key="a", label="直接回答", value="", success=False)}
        out = filler.render(tmpl, stats)
        assert "（直接回答暂缺）" in out
        assert re.search(r"（.*暂缺）", out)

    def test_unattempted_slot_no_raw_key(self):
        filler = SlotFiller(llm=StubLLM())
        tmpl = _tmpl(
            slots=[SlotDef("a", "甲", prompt="p"), SlotDef("b", "缺失槽", prompt="p")],
            phases=[["a", "b"]], render="{a} | {b}",
        )
        stats = FillStats(template_name="test", total_slots=2)
        stats.slot_results = {"a": SlotResult(key="a", label="甲", value="有值")}
        out = filler.render(tmpl, stats)
        assert "{b}" not in out
        assert "（缺失槽暂缺）" in out


# ===================== 失败兜底 =====================

class TestFailureFallback:
    def test_llm_failure_uses_default(self):
        stub = StubLLM(fail=True)
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl(
            [SlotDef("a", "甲", prompt="输出甲", default="默认值")], [["a"]], "{a}")
        stats = filler.fill(tmpl, query="测试-失败", context_docs=[])
        r = stats.slot_results["a"]
        assert r.value == "默认值"
        assert r.score == 0.1
        assert r.error
        assert stats.filled_slots == 1  # default 非空 → 视为已填充


# ===================== 统计 =====================

class TestStats:
    def test_parallel_gain_range(self):
        stub = StubLLM()
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl(
            slots=[SlotDef(f"s{i}", f"槽{i}", prompt=f"输出槽{i}") for i in range(3)],
            phases=[["s0", "s1", "s2"]], render="{s0}{s1}{s2}",
        )
        stats = filler.fill(tmpl, query="测试-增益", context_docs=[])
        assert 0.0 <= stats.parallel_gain <= 1.0

    def test_stream_ttft_measured(self):
        stub = StreamStubLLM()
        filler = SlotFiller(llm=stub)
        tmpl = _tmpl([SlotDef("a", "甲", prompt="输出甲")], [["a"]], "{a}")
        stats = filler.fill(tmpl, query="测试-流式", context_docs=[])
        assert stats.slot_results["a"].ttft_ms > 0
        assert stats.avg_ttft_ms > 0


# ===================== prompt 分层 =====================

class TestPromptLayering:
    def test_default_system_from_prompts(self):
        filler = SlotFiller(llm=StubLLM())
        assert filler.system_prompt == prompts.SLOT_FILLER_SYSTEM

    def test_custom_system_override(self):
        filler = SlotFiller(llm=StubLLM(), system_prompt="自定义约束")
        assert filler.system_prompt == "自定义约束"