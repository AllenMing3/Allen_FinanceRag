"""
研判 prompt 质量三修复 — 截断标记 / 输出校验 / 降级诚实化。

病根：
  1. text[:3000] 硬截断无标记 → LLM 把残文当全文，"信息不足"误判不可见
  2. rubric/verdict 一致性零程序化校验 → 非法 verdict、编造 evidence 静默流过
  3. 降级解析器伪造结构化数据 → impact 四维复制 verdict、severity 固定3、
     watch_next 固定套话，假数据冒充真分析

契约：
  - _truncate_for_prompt: >3000 字截断并追加标记；≤3000 恒等
  - _validate_assessment: 硬错（verdict/confidence 非法枚举）→ 重试一次
    （use_cache=False 防缓存命中同一错误响应），仍失败走 free-text 降级；
    软错（evidence 不在原文/severity 越界/条数超界）→ stats.validation_warnings
  - 降级 structured: degraded=True，无 impact/watch_next 伪造字段，
    severity 缺失不填假值，无占位信号/套话风险
  - stats 新增 validation_warnings / input_truncated（只增不改）

离线约束：_StubCaller 替代 LLMCaller（同 test_analysis_provenance 模式），
不碰真实 LLM/网络。

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定
"""
from types import SimpleNamespace

import pytest

from financial_rag.services import analysis as svc

pytestmark = pytest.mark.unit


# ===================== LLMCaller 桩（支持多次调用序列） =====================

class _StubCaller:
    """call_json 按 json_results 序列逐次返回；记录每次收到的 user prompt。"""

    json_results = []
    json_error = None
    text_content = ""
    text_error = None
    prompts = []
    json_calls = 0

    def __init__(self, llm):
        pass

    def call_json(self, user, **kwargs):
        _StubCaller.prompts.append(user)
        _StubCaller.json_calls += 1
        if self.json_error:
            raise self.json_error
        idx = min(_StubCaller.json_calls - 1, len(self.json_results) - 1)
        data = self.json_results[idx] if self.json_results else None
        if kwargs.get("return_meta"):
            meta = {"elapsed_ms": 0.0, "attempts": 1,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    "finish_reason": "stop", "cache_hit": False, "model": "stub"}
            return data, meta
        return data

    def call(self, user, **kwargs):
        if self.text_error:
            raise self.text_error
        return SimpleNamespace(content=self.text_content, usage={},
                               finish_reason="stop", model="stub")


@pytest.fixture
def stub_caller(monkeypatch):
    _StubCaller.json_results = []
    _StubCaller.json_error = None
    _StubCaller.text_content = ""
    _StubCaller.text_error = None
    _StubCaller.prompts = []
    _StubCaller.json_calls = 0
    monkeypatch.setattr("financial_rag.llm.caller.LLMCaller", _StubCaller)
    return _StubCaller


def _valid_data(**over):
    """合法主路径输出基线（evidence 逐字来自 _NEWS_TEXT）。"""
    data = {
        "verdict": "bullish", "confidence": "high",
        "verdict_reasoning": "营收增长30%支撑利好",
        "key_signals": [
            {"signal": "营收增长", "severity": 4, "type": "positive",
             "evidence": "营收增长30%"},
            {"signal": "订单充足", "severity": 3, "type": "positive",
             "evidence": "订单额超50亿美元"},
        ],
        "analysis": "分析正文",
    }
    data.update(over)
    return data


_NEWS_TEXT = "英伟达发布新一代芯片，营收增长30%，订单额超50亿美元。"
_FREETEXT = "【判断】利空\n【置信度】高\n【关键信号】\n- 订单下滑\n【影响分析】\n因果链正文"


# ======================================================================
# ① 截断标记：超长输入不再静默硬切
# ======================================================================

class TestTruncationMarker:
    def test_short_text_identity(self):
        assert svc._truncate_for_prompt("短文本") == "短文本"

    def test_boundary_3000_identity(self):
        text = "字" * 3000
        assert svc._truncate_for_prompt(text) == text

    def test_long_text_has_marker(self):
        out = svc._truncate_for_prompt("字" * 4000)
        assert out.startswith("字" * 3000)
        assert "截断" in out[3000:]

    def test_news_prompt_carries_marker(self, stub_caller):
        stub_caller.json_results = [_valid_data()]
        svc.analyze_news_text("长" * 4000 + _NEWS_TEXT, llm=object(),
                              retriever=None, kb_built=False)
        assert "截断" in stub_caller.prompts[0]

    def test_news_stats_input_truncated(self, stub_caller):
        stub_caller.json_results = [_valid_data()]
        r = svc.analyze_news_text("长" * 4000, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["assessment_stats"]["input_truncated"] is True

    def test_short_input_not_truncated(self, stub_caller):
        stub_caller.json_results = [_valid_data()]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["assessment_stats"]["input_truncated"] is False
        assert "截断" not in stub_caller.prompts[0]


# ======================================================================
# ②④ 输出校验：硬错重试 + 软错警告（回归锁定）
# ======================================================================

class TestValidationHardRetry:
    def test_illegal_verdict_retries_then_succeeds(self, stub_caller):
        stub_caller.json_results = [
            _valid_data(verdict="maybe"),   # 硬错 → 重试
            _valid_data(),                  # 第二次合法
        ]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["analysis_path"] == "llm_json"
        assert r["assessment"] == "bullish"
        assert stub_caller.json_calls == 2
        assert r["assessment_stats"]["attempts"] == 2

    def test_illegal_verdict_twice_falls_to_freetext(self, stub_caller):
        stub_caller.json_results = [_valid_data(verdict="maybe")]
        stub_caller.text_content = _FREETEXT
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["analysis_path"] == "llm_freetext"
        assert r["assessment"] == "bearish"
        assert stub_caller.json_calls == 2   # 重试有界：最多 2 次 JSON 调用

    def test_illegal_confidence_is_hard_error(self, stub_caller):
        stub_caller.json_results = [_valid_data(confidence="very-high")]
        stub_caller.text_content = _FREETEXT
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["analysis_path"] == "llm_freetext"

    def test_retry_tokens_accumulated(self, stub_caller):
        stub_caller.json_results = [_valid_data(verdict="x"), _valid_data()]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["assessment_stats"]["tokens"]["total_tokens"] == 30  # 15×2


class TestValidationSoftWarnings:
    def test_fabricated_evidence_warns_not_retries(self, stub_caller):
        stub_caller.json_results = [_valid_data(key_signals=[
            {"signal": "编造信号", "severity": 4, "type": "positive",
             "evidence": "原文中不存在的话"},
        ])]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["analysis_path"] == "llm_json"      # 软错不重试
        assert stub_caller.json_calls == 1
        warns = r["assessment_stats"]["validation_warnings"]
        assert any("evidence" in w for w in warns)

    def test_kb_evidence_prefix_exempt(self, stub_caller):
        stub_caller.json_results = [_valid_data(key_signals=[
            {"signal": "KB信号", "severity": 3, "type": "positive",
             "evidence": "知识库: 某背景事实"},
            {"signal": "订单充足", "severity": 3, "type": "positive",
             "evidence": "订单额超50亿美元"},
        ])]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["assessment_stats"]["validation_warnings"] == []

    def test_empty_evidence_exempt(self, stub_caller):
        data = _valid_data()
        data["key_signals"][0]["evidence"] = ""   # prompt 允许无依据填空
        stub_caller.json_results = [data]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["assessment_stats"]["validation_warnings"] == []

    def test_severity_out_of_range_warns(self, stub_caller):
        data = _valid_data()
        data["key_signals"][0]["severity"] = 9
        stub_caller.json_results = [data]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["analysis_path"] == "llm_json"
        assert any("severity" in w for w in r["assessment_stats"]["validation_warnings"])

    def test_valid_output_zero_warnings(self, stub_caller):
        stub_caller.json_results = [_valid_data()]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["assessment_stats"]["validation_warnings"] == []


# ======================================================================
# ③④ 降级诚实化：不伪造结构化数据（回归锁定）
# ======================================================================

class TestDegradedHonesty:
    def test_fallback_marks_degraded(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = _FREETEXT
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert r["structured"]["degraded"] is True

    def test_fallback_no_fake_impact(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = _FREETEXT
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert "impact" not in r["structured"]   # 不再复制 verdict 伪造四维

    def test_fallback_no_boilerplate_watch_next(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = _FREETEXT
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert not r["structured"].get("watch_next")

    def test_fallback_no_placeholder_signal(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = "【判断】中性\n【影响分析】正文无信号段"
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        sigs = r["structured"]["key_signals"]
        assert all(s["signal"] != "见分析正文" for s in sigs)

    def test_fallback_signal_no_fake_severity(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = _FREETEXT
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        for sig in r["structured"]["key_signals"]:
            assert sig.get("severity") is None   # 未解析出就不填假值

    def test_fallback_no_boilerplate_risk(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = "【判断】中性\n【影响分析】正文无风险段"
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert "信息不足，需进一步观察" not in r["structured"]["risks"]

    def test_json_path_not_degraded(self, stub_caller):
        stub_caller.json_results = [_valid_data()]
        r = svc.analyze_news_text(_NEWS_TEXT, llm=object(),
                                  retriever=None, kb_built=False)
        assert not r["structured"].get("degraded")


# ======================================================================
# ③ 一致性：stats 新字段全路径存在（只增不改）
# ======================================================================

class TestStatsContract:
    def test_rule_path_has_new_fields(self):
        r = svc.analyze_news_text(_NEWS_TEXT, llm=None, retriever=None, kb_built=False)
        stats = r["assessment_stats"]
        assert stats["validation_warnings"] == []
        assert stats["input_truncated"] is False

    def test_topic_no_llm_has_new_fields(self, monkeypatch):
        import importlib
        config_mod = importlib.import_module("financial_rag.config")
        mock_mod = importlib.import_module("financial_rag.mock_data")
        monkeypatch.setattr(config_mod, "is_mock_enabled", lambda: True)
        monkeypatch.setattr(
            mock_mod, "mock_search_news",
            lambda topic, max_news=20: {"items": [
                {"title": "校验契约测试-话题", "content": "订单增长",
                 "source": "测试社", "publish_time": "2026-11-21"},
            ]})
        r = svc.analyze_topic_research("校验契约话题", llm=None,
                                       retriever=None, kb_built=False)
        stats = r["assessment_stats"]
        assert stats["validation_warnings"] == []
        assert stats["input_truncated"] is False

    def test_topic_fallback_degraded(self, stub_caller, monkeypatch):
        import importlib
        config_mod = importlib.import_module("financial_rag.config")
        mock_mod = importlib.import_module("financial_rag.mock_data")
        monkeypatch.setattr(config_mod, "is_mock_enabled", lambda: True)
        monkeypatch.setattr(
            mock_mod, "mock_search_news",
            lambda topic, max_news=20: {"items": [
                {"title": "降级诚实话题", "content": "订单增长",
                 "source": "测试社", "publish_time": "2026-11-21"},
            ]})
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = "【判断】利好\n【置信度】中\n【综合分析】话题正文"
        r = svc.analyze_topic_research("降级诚实话题", llm=object(),
                                       retriever=None, kb_built=False)
        assert r["analysis_path"] == "llm_freetext"
        assert r["structured"]["degraded"] is True
        assert r["structured"]["investment_implication"] != "配置 API Key 获取投资建议"