"""
自主检索节点测试 — ToolCallSession 行为锁 + _agentic_search_kb 决策循环

离线确定性（无 API / 无网络 / 无 Phoenix）：
- 第一组: ToolCallSession 引擎行为锁（特征化，此前零覆盖）——轮次语义/预算/未知工具
- 第二组: _agentic_search_kb 准入/换词重查/去重/阈值/预算/降级/0命中可观测
- 第三组: 深度分析主路径集成（llm=None 降级回归 + agentic 端到端同源契约）

stub LLM 只测接线行为（决策循环组装/证据收割），不评估生成质量
（沿用 test_slot_filler StubLLM 先例）；ScriptedRetriever 只 mock KB 数据源，
不 mock 检索算法本身。真实决策质量需配 API key 实测。

运行: python -m pytest tests/test_agentic_retrieval.py -v
"""
import json
from types import SimpleNamespace

import pytest

from financial_rag.services.analysis import _search_kb, analyze_news_text
from financial_rag.tools.core import (
    FunctionDef, FunctionRegistry, ToolCallSession,
)

# 业务主链行为锁 + 降级静默正确性 → 整文件 critical（判定标准①③，见 pytest.ini）
pytestmark = [pytest.mark.unit, pytest.mark.critical]


# ===================== 确定性 stub =====================

class FCStubLLM:
    """按轮次脚本化 tool_calls：每元素是 query 列表（发起检索）或 None（收工）。

    chat() 返回 canned 研判 JSON，供第三组端到端走 LLMCaller.call_json 主路径。
    """

    def __init__(self, rounds, tool_name="search_financial_data", arg_name="query",
                 assessment_json=None):
        self.rounds = rounds
        self.tool_name = tool_name
        self.arg_name = arg_name
        self.i = 0
        self.fc_calls = []
        self.chat_calls = []
        self.assessment_json = assessment_json or {
            "verdict": "neutral", "confidence": "low", "analysis": "stub 研判",
        }

    def chat_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.fc_calls.append(messages)
        spec = self.rounds[min(self.i, len(self.rounds) - 1)]
        self.i += 1
        if spec is None:
            return SimpleNamespace(content="证据收集完成。", tool_calls=[])
        tcs = [
            {"id": f"call_{self.i}_{j}", "type": "function",
             "function": {"name": self.tool_name,
                          "arguments": json.dumps({self.arg_name: q}, ensure_ascii=False)}}
            for j, q in enumerate(spec)
        ]
        return SimpleNamespace(content="", tool_calls=tcs)

    def chat(self, messages, system=None, **kwargs):
        self.chat_calls.append(messages)
        content = json.dumps(self.assessment_json, ensure_ascii=False)
        return SimpleNamespace(content=content, usage={"total_tokens": len(content)})


class ExplodingFCStub(FCStubLLM):
    """FC 通道抛异常 — 验证 _agentic_search_kb 单发兜底。"""

    def __init__(self):
        super().__init__(rounds=[None])

    def chat_with_tools(self, *args, **kwargs):
        raise RuntimeError("FC 通道故障")


def _hit(text, score=0.9, source="kb_test"):
    """检索结果项（与 HybridRetriever 输出同构）"""
    return {"text": text, "meta": {"source": source}, "score": score}


class ScriptedRetriever:
    """脚本化 KB 数据源：query 子串匹配 → 预设结果（只 mock 数据源）。"""

    def __init__(self, by_query=None, default=None):
        self.by_query = by_query or {}
        self.default = default if default is not None else []
        self.queries = []
        self.last_gate_info = None

    def _results(self, query):
        for kw, res in self.by_query.items():
            if kw in query:
                return list(res)
        return list(self.default)

    def search(self, query, top_k=5, **kwargs):
        self.queries.append(query)
        return self._results(query)[:top_k]

    def search_with_scores(self, query, top_k=5, **kwargs):
        self.queries.append(query)
        return self._results(query)[:top_k], None


def _echo_registry():
    reg = FunctionRegistry(name="echo_test")
    reg.add(FunctionDef(
        name="echo", description="回显输入",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}}, "required": []},
        callback=lambda text="": {"echo": text},
    ))
    return reg


# ===================== 第一组: ToolCallSession 引擎行为锁 =====================

class TestToolCallSessionBehaviorLock:
    """特征化锁定（此前零覆盖）：轮次语义 / 预算强制 / 未知工具不中断。"""

    def test_single_round_then_stop(self):
        stub = FCStubLLM([["hello"], None], tool_name="echo", arg_name="text")
        session = ToolCallSession(llm=stub, registry=_echo_registry(), max_rounds=5)
        stats = session.run("回显测试")
        assert stats.rounds == 1
        assert len(stats.calls) == 1
        assert stats.succeeded == 1
        assert stats.calls[0].result == {"echo": "hello"}
        # 收工轮也发了一次 LLM 请求（无 tool_call → break）
        assert len(stub.fc_calls) == 2

    def test_budget_enforced(self):
        stub = FCStubLLM([["a"], ["b"], ["c"]], tool_name="echo", arg_name="text")
        session = ToolCallSession(llm=stub, registry=_echo_registry(), max_rounds=2)
        stats = session.run("预算测试")
        assert stats.rounds == 2
        assert len(stats.calls) == 2
        assert len(stub.fc_calls) == 2

    def test_unknown_tool_does_not_break_session(self):
        stub = FCStubLLM([["x"], None], tool_name="no_such_tool")
        session = ToolCallSession(llm=stub, registry=_echo_registry(), max_rounds=3)
        stats = session.run("未知工具")
        assert stats.calls[0].success is False
        assert "未知能力" in stats.calls[0].error
        assert stats.rounds == 1  # 失败调用也计入轮次，会话正常收工


# ===================== 第二组: _agentic_search_kb 决策循环 =====================

class TestAgenticSearchKB:

    def test_llm_none_fallback_identical(self):
        """降级契约：llm=None 输出与 _search_kb 逐字段一致（不附加 trace）。"""
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(default=[_hit("茅台2024年营收1741亿元。")])
        sources, info = _agentic_search_kb(None, r, "茅台营收", True)
        exp_sources, exp_info = _search_kb(r, "茅台营收", True)
        assert sources == exp_sources
        assert info == exp_info

    def test_mock_mode_fallback(self, monkeypatch):
        from financial_rag.config import config
        from financial_rag.services.analysis import _agentic_search_kb
        monkeypatch.setattr(config.mock, "enable", True)
        stub = FCStubLLM([["茅台"], None])
        r = ScriptedRetriever(default=[_hit("证据")])
        sources, info = _agentic_search_kb(stub, r, "茅台", True)
        assert stub.fc_calls == []  # mock 模式不进循环
        assert sources
        assert "agentic_trace" not in info

    def test_requery_on_empty_then_hit(self):
        """核心 agent 行为：第一轮空结果 → LLM 换词重查 → 命中收割。"""
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(
            by_query={"白酒": [_hit("茅台是白酒龙头，2024年营收1741亿元。")]},
            default=[],
        )
        stub = FCStubLLM([["茅台"], ["白酒"], None])
        sources, info = _agentic_search_kb(stub, r, "茅台", True)
        assert len(sources) == 1
        assert "1741亿" in sources[0]["text"]
        trace = info["agentic_trace"]
        assert trace["enabled"] is True
        assert trace["stop_reason"] == "llm_stop"
        assert len(trace["rounds"]) == 2
        assert trace["rounds"][0]["queries"] == ["茅台"]
        assert trace["rounds"][0]["total_found"] == 0
        assert trace["rounds"][1]["total_found"] == 1
        assert r.queries == ["茅台", "白酒"]

    def test_dedup_across_rounds(self):
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(default=[_hit("同一条证据文本")])
        stub = FCStubLLM([["查询A"], ["查询B"], None])
        sources, info = _agentic_search_kb(stub, r, "初始", True)
        assert len(sources) == 1
        assert info["above_threshold"] == 1

    def test_threshold_filter(self):
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(default=[_hit("低分", score=0.2), _hit("高分", score=0.9)])
        stub = FCStubLLM([["q"], None])
        sources, info = _agentic_search_kb(stub, r, "q", True)
        assert [s["text"] for s in sources] == ["高分"]
        assert info["above_threshold"] == 1
        assert info["total_results"] == 2  # 原始结果含低分项

    def test_budget_stop(self):
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(default=[])
        stub = FCStubLLM([["q1"], ["q2"], ["q3"], ["q4"]])
        sources, info = _agentic_search_kb(stub, r, "q", True, max_rounds=2)
        trace = info["agentic_trace"]
        assert trace["stop_reason"] == "budget"
        assert len(trace["rounds"]) == 2
        assert len(stub.fc_calls) == 2

    def test_exception_fallback(self):
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(default=[_hit("兜底证据")])
        stub = ExplodingFCStub()
        sources, info = _agentic_search_kb(stub, r, "初始词", True)
        assert sources[0]["text"] == "兜底证据"
        assert info["agentic_trace"]["stop_reason"] == "fallback"
        assert r.queries == ["初始词"]  # 单发兜底用初始 query

    def test_zero_hit_full_diagnostics(self):
        """0 命中可观测：sources 空但每轮诊断完整（硬拦截+全量反馈规范）。"""
        from financial_rag.services.analysis import _agentic_search_kb
        r = ScriptedRetriever(default=[])
        stub = FCStubLLM([["词A"], ["词B"], None])
        sources, info = _agentic_search_kb(stub, r, "词A", True)
        assert sources == []
        trace = info["agentic_trace"]
        assert len(trace["rounds"]) == 2
        assert all(rd["total_found"] == 0 for rd in trace["rounds"])
        assert trace["rounds"][0]["queries"] == ["词A"]
        assert info["total_results"] == 0

    def test_kb_not_built_skips_loop(self):
        from financial_rag.services.analysis import _agentic_search_kb
        stub = FCStubLLM([["q"], None])
        r = ScriptedRetriever(default=[_hit("x")])
        sources, info = _agentic_search_kb(stub, r, "q", False)
        assert sources == []
        assert info["reason"] == "KB 未构建"
        assert stub.fc_calls == []


# ===================== 第三组: 深度分析主路径集成 =====================

class TestDeepAnalysisIntegration:

    def test_analyze_news_llm_none_regression(self, ai_news_text):
        """降级不破坏主链：llm=None 输出结构完整，检索走单发。"""
        r = ScriptedRetriever(default=[_hit("Blackwell 背景证据", score=0.9)])
        result = analyze_news_text(ai_news_text, llm=None, retriever=r, kb_built=True)
        assert set(result) >= {
            "assessment", "analysis", "structured",
            "kb_sources", "kb_grounding", "kb_search_info",
        }
        assert result["kb_sources"][0]["text"] == "Blackwell 背景证据"
        assert "agentic_trace" not in result["kb_search_info"]

    def test_analyze_news_agentic_end_to_end(self, ai_news_text):
        """agentic 端到端：trace 透出 + grounding 与 sources 同源（ContextBuilder 未被绕过）。"""
        r = ScriptedRetriever(
            by_query={"GPU": [_hit("Blackwell GPU 训练性能较 H100 提升 4 倍。")]},
            default=[],
        )
        stub = FCStubLLM([["Blackwell"], ["GPU"], None])
        result = analyze_news_text(ai_news_text, llm=stub, retriever=r, kb_built=True)
        info = result["kb_search_info"]
        assert info["agentic_trace"]["enabled"] is True
        assert len(info["agentic_trace"]["rounds"]) == 2
        # 唯一真相源契约：喂 LLM 的 grounding 与 Guard 所见 sources 同源
        assert [g["text"] for g in result["kb_grounding"]] == \
               [s["text"] for s in result["kb_sources"]]
        # stub 研判 JSON 走通 LLMCaller.call_json 主路径
        assert result["structured"].get("verdict") == "neutral"