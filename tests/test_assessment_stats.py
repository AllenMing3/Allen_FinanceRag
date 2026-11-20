"""
研判消耗可观测性测试 — 利好利空研判的耗时/token/重试/截断全透出。

病根：DashScopeLLM 的 usage/finish_reason 一直都在，但 call_json 只返回
解析后的 dict，元数据半路丢弃 → 重试放大成本不可见、max_tokens 截断不可见。

契约：
  - call_json(return_meta=True) → (data, meta)，默认行为不变（8 处调用方零影响）
  - meta: elapsed_ms / attempts / usage 累加 / finish_reason / cache_hit / model
  - _llm_news/_llm_topic_assessment 返回 5 元组，第 5 位 stats（path 同 analysis_path）
  - analyze_news_text / analyze_topic_research 返回 assessment_stats（只增不改）
  - truncated = finish_reason == "length"（截断铁证，不再静默）

离线约束：StubLLM 经真实 LLMCaller（进程级缓存 → 各用例 prompt 唯一防串扰）；
不碰真实 LLM/网络。

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定
"""
import json
from types import SimpleNamespace

import pytest

from financial_rag.llm.caller import LLMCaller
from financial_rag.services import analysis as svc


# ===================== StubLLM =====================

class StubLLM:
    """按调用次序返回预设响应；记录调用次数。"""

    def __init__(self, contents, usage=None, finish_reason="stop"):
        self.contents = list(contents)
        self.usage = usage or {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        self.finish_reason = finish_reason
        self.calls = 0

    def chat(self, messages, system=None, **kwargs):
        self.calls += 1
        content = self.contents[min(self.calls - 1, len(self.contents) - 1)]
        return SimpleNamespace(
            content=content, model="stub-model", usage=dict(self.usage),
            finish_reason=self.finish_reason, tool_calls=[])


_VALID_JSON = json.dumps({
    "verdict": "bullish", "confidence": "high",
    "verdict_reasoning": "营收增长支撑利好", "analysis": "分析正文",
    "key_signals": [{"signal": "营收增长30%", "severity": 4, "type": "positive",
                     "evidence": "营收增长30%"}],
}, ensure_ascii=False)

_FREETEXT = "【判断】利空\n【置信度】高\n【关键信号】\n- 订单下滑\n【影响分析】\n因果链正文"


# ======================================================================
# ①② call_json 元数据：默认契约不变 + return_meta 全量诊断
# ======================================================================

class TestCallJsonMeta:
    pytestmark = pytest.mark.unit

    def test_default_behavior_unchanged(self):
        """8 处既有调用方零影响：不传 return_meta 仍只返回 dict。"""
        caller = LLMCaller(StubLLM([_VALID_JSON]), cache_ttl=0)
        data = caller.call_json("元数据测试-默认契约", system="输出JSON")
        assert isinstance(data, dict)
        assert data["verdict"] == "bullish"

    @pytest.mark.critical
    def test_return_meta_fields(self):
        caller = LLMCaller(StubLLM([_VALID_JSON]), cache_ttl=0)
        data, meta = caller.call_json(
            "元数据测试-字段齐全", system="输出JSON", return_meta=True)
        assert data["verdict"] == "bullish"
        assert meta["attempts"] == 1
        assert meta["elapsed_ms"] >= 0
        assert meta["usage"]["total_tokens"] == 150
        assert meta["usage"]["input_tokens"] == 100
        assert meta["finish_reason"] == "stop"
        assert meta["cache_hit"] is False
        assert meta["model"] == "stub-model"

    @pytest.mark.critical
    @pytest.mark.regression
    def test_retry_cost_visible(self):
        """BUG: JSON 解析失败重试最多 3 次，token 成本此前完全不可见。
        修复后 attempts 与 usage 累加必须如实反映重试消耗。"""
        stub = StubLLM(["这不是JSON", _VALID_JSON])
        caller = LLMCaller(stub, cache_ttl=0)
        data, meta = caller.call_json(
            "元数据测试-重试成本", system="输出JSON", return_meta=True)
        assert data is not None
        assert meta["attempts"] == 2
        assert stub.calls == 2
        assert meta["usage"]["total_tokens"] == 300   # 两次调用累加

    def test_all_retries_exhausted_returns_none_with_meta(self):
        stub = StubLLM(["坏JSON1", "坏JSON2", "坏JSON3"])
        caller = LLMCaller(stub, cache_ttl=0)
        data, meta = caller.call_json(
            "元数据测试-重试耗尽", system="输出JSON", return_meta=True)
        assert data is None
        assert meta["attempts"] == 3
        assert meta["usage"]["total_tokens"] == 450

    def test_cache_hit_observable(self):
        """缓存命中耗时骤降/token 为 0 — 不标注会误导性能判断。"""
        stub = StubLLM([_VALID_JSON])
        caller = LLMCaller(stub, cache_ttl=300)
        prompt = "元数据测试-缓存命中观测"
        caller.call_json(prompt, system="输出JSON")
        data, meta = caller.call_json(prompt, system="输出JSON", return_meta=True)
        assert data is not None
        assert meta["cache_hit"] is True
        assert stub.calls == 1, "第二次应命中缓存不再调 LLM"

    def test_truncation_finish_reason_surfaced(self):
        """max_tokens 截断的铁证 finish_reason=length 必须透出。"""
        caller = LLMCaller(
            StubLLM([_VALID_JSON], finish_reason="length"), cache_ttl=0)
        _, meta = caller.call_json(
            "元数据测试-截断信号", system="输出JSON", return_meta=True)
        assert meta["finish_reason"] == "length"


# ======================================================================
# ①③ 研判函数：5 元组 stats 与 analysis_path 同源一致
# ======================================================================

class TestAssessmentStats:
    pytestmark = pytest.mark.unit

    @pytest.mark.critical
    def test_news_json_path_stats(self):
        stub = StubLLM([_VALID_JSON])
        verdict, structured, conf, path, stats = svc._llm_news_assessment(
            stub, "研判stats测试-新闻JSON路径：营收增长30%", "news", {}, {}, "")
        assert verdict == "bullish"
        assert path == "llm_json"
        assert stats["path"] == path               # 同源一致
        assert stats["tokens"]["total_tokens"] == 150
        assert stats["attempts"] == 1
        assert stats["truncated"] is False
        assert stats["elapsed_ms"] >= 0

    def test_news_freetext_path_stats_accumulates(self):
        """主路径 3 次重试全败 + 降级 1 次 = 4 次调用，成本必须全算进去。"""
        stub = StubLLM(["坏JSON", _FREETEXT])
        verdict, structured, conf, path, stats = svc._llm_news_assessment(
            stub, "研判stats测试-新闻降级路径：订单下滑", "news", {}, {}, "")
        assert path == "llm_freetext"
        assert verdict == "bearish"
        assert stats["attempts"] == 4              # 3 次 JSON 重试 + 1 次降级
        assert stats["tokens"]["total_tokens"] == 600
        assert stub.calls == 4

    def test_news_truncated_flag(self):
        stub = StubLLM([_VALID_JSON], finish_reason="length")
        _, _, _, path, stats = svc._llm_news_assessment(
            stub, "研判stats测试-截断标记：订单增长", "news", {}, {}, "")
        assert path == "llm_json"
        assert stats["truncated"] is True

    def test_topic_json_path_stats(self):
        stub = StubLLM([_VALID_JSON])
        verdict, structured, conf, path, stats = svc._llm_topic_assessment(
            stub, "研判stats测试-话题JSON", [], "话题正文", "")
        assert path == "llm_json"
        assert stats["tokens"]["total_tokens"] == 150
        assert stats["path"] == "llm_json"


# ======================================================================
# ①③ 端到端透出：assessment_stats 只增不改 + 无 LLM 路径也有诊断
# ======================================================================

class TestEndToEndStats:
    pytestmark = pytest.mark.integration

    def test_news_rule_path_has_stats(self):
        result = svc.analyze_news_text(
            "端到端stats测试：英伟达营收增长30%，订单额超50亿美元。",
            llm=None, retriever=None, kb_built=False)
        assert result["analysis_path"] == "rule"
        stats = result["assessment_stats"]
        assert stats["path"] == "rule"
        assert stats["elapsed_ms"] >= 0
        assert stats["tokens"]["total_tokens"] == 0   # 规则路径零 token
        # 既有契约字段不受影响（只增不改）
        assert "assessment" in result and "structured" in result

    def test_news_llm_path_stats_passthrough(self):
        stub = StubLLM([_VALID_JSON])
        result = svc.analyze_news_text(
            "端到端stats测试-LLM路径：英伟达发布新芯片，订单增长。",
            llm=stub, retriever=None, kb_built=False)
        assert result["analysis_path"] == "llm_json"
        stats = result["assessment_stats"]
        assert stats["path"] == "llm_json"
        assert stats["tokens"]["total_tokens"] == 150

    def test_topic_no_llm_path_has_stats(self, monkeypatch):
        import importlib
        config_mod = importlib.import_module("financial_rag.config")
        mock_mod = importlib.import_module("financial_rag.mock_data")
        monkeypatch.setattr(config_mod, "is_mock_enabled", lambda: True)
        monkeypatch.setattr(
            mock_mod, "mock_search_news",
            lambda topic, max_news=20: {"items": [
                {"title": "端到端stats测试-话题", "content": "订单增长",
                 "source": "测试社", "publish_time": "2026-11-21"},
            ]})
        result = svc.analyze_topic_research(
            "端到端stats话题", llm=None, retriever=None, kb_built=False)
        assert result["analysis_path"] == "no_llm"
        stats = result["assessment_stats"]
        assert stats["path"] == "no_llm"
        assert stats["elapsed_ms"] >= 0


# ======================================================================
# ④ 回归锁定：SDK usage 是 __slots__ 对象，__dict__ 永远为空
# ======================================================================

class TestUsageConversionRegression:
    """病根：dashscope 2.x 的 GenerationUsage 用 __slots__ 存数据，
    resp.usage.__dict__ 拿到空 dict → token 统计永远为 0。
    修复：dict(resp.usage)。本锁防止改回 __dict__ 写法。
    """
    pytestmark = pytest.mark.unit

    def test_slots_usage_survives_chat(self, monkeypatch):
        import dashscope
        from financial_rag.llm.dashscope_client import DashScopeLLM

        class SlotsUsage:
            # 模拟真实 GenerationUsage：无 __dict__，仅支持 dict() 协议
            __slots__ = ("_data",)

            def __init__(self, data):
                self._data = data

            def keys(self):
                return self._data.keys()

            def __getitem__(self, key):
                return self._data[key]

        fake_resp = SimpleNamespace(
            status_code=200,
            usage=SlotsUsage({"input_tokens": 9, "output_tokens": 5, "total_tokens": 14}),
            output=SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok"), finish_reason="stop")]),
        )
        monkeypatch.setattr(dashscope.Generation, "call",
                            staticmethod(lambda **kw: fake_resp))
        llm = DashScopeLLM(api_key="test-key")
        resp = llm.chat("hi")
        assert resp.usage["input_tokens"] == 9
        assert resp.usage["output_tokens"] == 5
        assert resp.usage["total_tokens"] == 14
        assert resp.finish_reason == "stop"