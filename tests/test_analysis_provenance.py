"""
分析结论可解释性 — 生成路径透出 + 出处声明

离线验证（无 LLM / 无网络）：
- analysis_path：llm_json / llm_freetext / llm_failed / rule / no_llm 五条路径取值
- _attach_provenance：模型/路径/KB 参考/时间结构、来源去重、cfg 缺失与异常降级

运行: python -m pytest tests/test_analysis_provenance.py -v
"""
from types import SimpleNamespace

import pytest

from financial_rag.services import analysis as svc

pytestmark = pytest.mark.unit


# ===================== LLMCaller 桩 =====================

class _StubCaller:
    """替代 LLMCaller：call_json / call 按预设返回或抛错"""

    json_result = None
    json_error = None
    text_content = ""
    text_error = None

    def __init__(self, llm):
        pass

    def call_json(self, user, **kwargs):
        if self.json_error:
            raise self.json_error
        return self.json_result

    def call(self, user, **kwargs):
        if self.text_error:
            raise self.text_error
        return SimpleNamespace(content=self.text_content)


@pytest.fixture
def stub_caller(monkeypatch):
    # 重置类属性，避免跨用例状态残留
    _StubCaller.json_result = None
    _StubCaller.json_error = None
    _StubCaller.text_content = ""
    _StubCaller.text_error = None
    monkeypatch.setattr("financial_rag.llm.caller.LLMCaller", _StubCaller)
    return _StubCaller


_NEWS_TEXT = "英伟达发布新一代芯片，营收增长30%，订单额超50亿美元。"


# ===================== analysis_path：新闻路径 =====================

class TestNewsAnalysisPath:
    def test_no_llm_is_rule(self):
        result = svc.analyze_news_text(_NEWS_TEXT, llm=None, retriever=None, kb_built=False)
        assert result["analysis_path"] == "rule"

    def test_json_success_is_llm_json(self, stub_caller):
        stub_caller.json_result = {
            "verdict": "bullish", "confidence": "high",
            "verdict_reasoning": "营收增长30%支撑利好",
            "key_signals": [{"signal": "营收增长", "severity": 4, "type": "positive",
                             "evidence": "营收增长30%"}],
            "analysis": "分析正文",
        }
        result = svc.analyze_news_text(_NEWS_TEXT, llm=object(), retriever=None, kb_built=False)
        assert result["analysis_path"] == "llm_json"
        assert result["structured"]["verdict_reasoning"] == "营收增长30%支撑利好"

    def test_json_fail_freetext_ok_is_llm_freetext(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_content = "【判断】利空\n【置信度】高\n【关键信号】\n- 订单下滑\n【影响分析】\n因果链正文"
        result = svc.analyze_news_text(_NEWS_TEXT, llm=object(), retriever=None, kb_built=False)
        assert result["analysis_path"] == "llm_freetext"
        assert result["assessment"] == "bearish"

    def test_both_fail_is_llm_failed(self, stub_caller):
        stub_caller.json_error = ValueError("bad json")
        stub_caller.text_error = RuntimeError("api down")
        result = svc.analyze_news_text(_NEWS_TEXT, llm=object(), retriever=None, kb_built=False)
        assert result["analysis_path"] == "llm_failed"
        assert result["assessment"] == "unknown"


# ===================== analysis_path：话题路径 =====================

@pytest.fixture
def mock_news(monkeypatch):
    """话题调研离线化：mock 开关 + mock 新闻源

    注：包级 financial_rag.config 属性是 AppConfig 实例（遮蔽同名子模块），
    字符串路径 monkeypatch 会解析错对象，须用 importlib 拿真实模块。
    """
    import importlib
    config_mod = importlib.import_module("financial_rag.config")
    mock_mod = importlib.import_module("financial_rag.mock_data")
    monkeypatch.setattr(config_mod, "is_mock_enabled", lambda: True)
    monkeypatch.setattr(
        mock_mod, "mock_search_news",
        lambda topic, max_news=20: {"items": [
            {"title": "英伟达芯片订单增长", "content": "订单额超50亿美元", "source": "测试社", "publish_time": "2026-11-19"},
        ]},
    )


class TestTopicAnalysisPath:
    def test_no_llm_is_no_llm(self, mock_news):
        result = svc.analyze_topic_research("英伟达", llm=None, retriever=None, kb_built=False)
        assert result["analysis_path"] == "no_llm"

    def test_json_success_is_llm_json(self, mock_news, stub_caller):
        stub_caller.json_result = {
            "verdict": "bullish", "confidence": "medium",
            "verdict_reasoning": "订单增长支撑利好",
            "sub_topics": [{"name": "芯片订单", "sentiment": "positive",
                            "summary": "订单增长", "evidence": "订单额超50亿美元"}],
            "analysis": "调研正文",
        }
        result = svc.analyze_topic_research("英伟达", llm=object(), retriever=None, kb_built=False)
        assert result["analysis_path"] == "llm_json"


# ===================== _attach_provenance：出处声明 =====================

class TestAttachProvenance:
    @pytest.fixture
    def router(self, monkeypatch):
        from financial_rag.api import analysis_router
        cfg = SimpleNamespace(llm=SimpleNamespace(model="qwen-test", provider="dashscope"))
        monkeypatch.setattr(analysis_router, "_state", {"cfg": cfg})
        return analysis_router

    def test_full_structure(self, router):
        result = {
            "analysis_path": "llm_json",
            "kb_sources": [
                {"text": "a", "source": "年报.pdf"},
                {"text": "b", "source": "年报.pdf"},
                {"text": "c", "source": "行业报告.pdf"},
            ],
        }
        router._attach_provenance(result)
        prov = result["provenance"]
        assert prov["model"] == "qwen-test"
        assert prov["provider"] == "dashscope"
        assert prov["path"] == "llm_json"
        assert prov["kb_hits"] == 3
        assert prov["kb_documents"] == ["年报.pdf", "行业报告.pdf"]  # 去重保序
        assert prov["generated_at"]

    def test_no_kb_sources(self, router):
        result = {"analysis_path": "rule", "kb_sources": []}
        router._attach_provenance(result)
        assert result["provenance"]["kb_hits"] == 0
        assert result["provenance"]["kb_documents"] == []

    def test_cfg_missing_degrades(self, monkeypatch):
        from financial_rag.api import analysis_router
        monkeypatch.setattr(analysis_router, "_state", {})
        result = {"kb_sources": [{"text": "a", "source": "x.pdf"}]}
        analysis_router._attach_provenance(result)
        prov = result["provenance"]
        assert prov["model"] == ""
        assert prov["kb_hits"] == 1  # KB 信息仍透出

    def test_exception_degrades_to_empty(self, monkeypatch):
        from financial_rag.api import analysis_router

        class _BrokenCfg:
            @property
            def llm(self):
                raise RuntimeError("cfg broken")

        monkeypatch.setattr(analysis_router, "_state", {"cfg": _BrokenCfg()})
        result = {}
        analysis_router._attach_provenance(result)
        assert result["provenance"] == {}


# ===================== Guard passed 透出回归锁 =====================

class TestGuardPassedPassthrough:
    """check_hallucination 丢失 passed 会让高分也显示"未通过"，与及格线解释矛盾"""

    def test_passed_matches_service(self):
        from financial_rag.tools.scoring_tools import check_hallucination
        from financial_rag.guard.service import ReliabilityService
        answer = "英伟达营收增长30%，订单额超50亿美元。"
        sources = [{"text": "据报道，英伟达营收增长30%，订单额超50亿美元。"}]
        out = check_hallucination(answer, sources, mode="analysis")
        ref = ReliabilityService().check(answer, sources, mode="analysis")
        assert "passed" in out
        assert out["passed"] == ref["passed"]