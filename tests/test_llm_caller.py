"""
Test LLMCaller — smart call layer covering retry, JSON parsing, caching, and constraints.
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict

from financial_rag.llm.caller import (
    LLMCaller,
    get_caller,
    parse_json_from_text,
    parse_json_list_from_text,
    DEFAULT_SYSTEM_CONSTRAINTS,
    JSON_OUTPUT_HINT,
)
from financial_rag.llm.dashscope_client import LLMResponse


# ===================== Mock LLM =====================

class MockLLM:
    """Mock DashScopeLLM — records calls, returns scripted responses."""

    def __init__(self, responses=None, fail_count=0):
        self.responses = list(responses) if responses else []
        self.fail_count = fail_count
        self.call_count = 0
        self.last_system = None
        self.last_messages = None
        self.last_kwargs = {}

    def chat(self, messages, system=None, **kwargs):
        self.call_count += 1
        self.last_system = system
        self.last_messages = messages
        self.last_kwargs = kwargs

        # Simulate failures
        if self.call_count <= self.fail_count:
            raise RuntimeError(f"Mock API error (attempt {self.call_count})")

        # Return next scripted response
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = "default response"

        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"total_tokens": 10},
            finish_reason="stop",
        )

    def chat_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.call_count += 1
        return LLMResponse(
            content="",
            model="mock-model",
            tool_calls=[{
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }],
        )


# ===================== JSON 解析 =====================

class TestParseJsonFromText:

    def test_bare_json(self):
        assert parse_json_from_text('{"key": "value"}') == {"key": "value"}

    def test_json_in_markdown_block(self):
        text = 'Some text\n```json\n{"a": 1}\n```\nMore text'
        assert parse_json_from_text(text) == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"answer": 42} done'
        assert parse_json_from_text(text) == {"answer": 42}

    def test_nested_json(self):
        text = '{"outer": {"inner": "value"}}'
        result = parse_json_from_text(text)
        assert result == {"outer": {"inner": "value"}}

    def test_empty_string(self):
        assert parse_json_from_text("") is None

    def test_no_json(self):
        assert parse_json_from_text("just plain text") is None

    def test_invalid_json(self):
        assert parse_json_from_text("{broken json}") is None

    def test_list_not_returned(self):
        # parse_json_from_text should NOT return lists
        assert parse_json_from_text("[1, 2, 3]") is None


class TestParseJsonListFromText:

    def test_bare_list(self):
        assert parse_json_list_from_text('[1, 2, 3]') == [1, 2, 3]

    def test_list_in_markdown(self):
        text = '```json\n["a", "b"]\n```'
        assert parse_json_list_from_text(text) == ["a", "b"]

    def test_single_object_becomes_list(self):
        text = '```json\n{"a": 1}\n```'
        result = parse_json_list_from_text(text)
        assert result == [{"a": 1}]

    def test_empty_string(self):
        assert parse_json_list_from_text("") is None

    def test_no_list(self):
        assert parse_json_list_from_text("plain text") is None


# ===================== LLMCaller.call =====================

class TestLLMCallerCall:

    def setup_method(self):
        LLMCaller._cache.clear()

    def test_basic_call(self):
        llm = MockLLM(responses=["Hello!"])
        caller = LLMCaller(llm)
        resp = caller.call("Hi", use_cache=False)
        assert resp.content == "Hello!"
        assert llm.call_count == 1

    def test_default_system_constraints(self):
        llm = MockLLM(responses=["OK"])
        caller = LLMCaller(llm)
        caller.call("test", use_cache=False)
        # When no system prompt provided, default should be applied
        assert llm.last_system == DEFAULT_SYSTEM_CONSTRAINTS

    def test_custom_system_overrides_constraints(self):
        llm = MockLLM(responses=["OK"])
        caller = LLMCaller(llm)
        caller.call("test", system="Custom system", use_cache=False)
        assert llm.last_system == "Custom system"

    def test_retry_on_failure(self):
        llm = MockLLM(responses=["Success"], fail_count=2)
        caller = LLMCaller(llm, max_retries=3)
        with patch("financial_rag.llm.caller.time.sleep"):  # skip actual sleep
            resp = caller.call("test", use_cache=False)
        assert resp.content == "Success"
        assert llm.call_count == 3  # 2 failures + 1 success

    def test_retry_exhausted(self):
        llm = MockLLM(fail_count=10)  # always fails
        caller = LLMCaller(llm, max_retries=2)
        with patch("financial_rag.llm.caller.time.sleep"):
            with pytest.raises(RuntimeError, match="重试耗尽"):
                caller.call("test", use_cache=False)
        assert llm.call_count == 3  # initial + 2 retries

    def test_kwargs_passthrough(self):
        llm = MockLLM(responses=["OK"])
        caller = LLMCaller(llm)
        caller.call("test", temperature=0.5, max_tokens=100, use_cache=False)
        assert llm.last_kwargs.get("temperature") == 0.5
        assert llm.last_kwargs.get("max_tokens") == 100


# ===================== LLMCaller.call_json =====================

class TestLLMCallerCallJson:

    def setup_method(self):
        LLMCaller._cache.clear()

    def test_call_json_dict(self):
        llm = MockLLM(responses=['{"key": "value"}'])
        caller = LLMCaller(llm)
        result = caller.call_json("test")
        assert result == {"key": "value"}

    def test_call_json_list(self):
        llm = MockLLM(responses=['["a", "b", "c"]'])
        caller = LLMCaller(llm)
        result = caller.call_json("test")
        assert result == ["a", "b", "c"]

    def test_call_json_with_surrounding_text(self):
        llm = MockLLM(responses=['Here is the answer: {"x": 1} done'])
        caller = LLMCaller(llm)
        result = caller.call_json("test")
        assert result == {"x": 1}

    def test_call_json_markdown_block(self):
        llm = MockLLM(responses=['```json\n{"result": true}\n```'])
        caller = LLMCaller(llm)
        result = caller.call_json("test")
        assert result == {"result": True}

    def test_call_json_retry_on_bad_json(self):
        # First response is garbage, second is valid
        llm = MockLLM(responses=["I cannot produce JSON", '{"valid": true}'])
        caller = LLMCaller(llm, max_retries=2)
        with patch("financial_rag.llm.caller.time.sleep"):
            result = caller.call_json("test")
        assert result == {"valid": True}
        assert llm.call_count == 2

    def test_call_json_returns_none_on_exhaustion(self):
        llm = MockLLM(responses=["garbage"] * 5)
        caller = LLMCaller(llm, max_retries=1)
        with patch("financial_rag.llm.caller.time.sleep"):
            result = caller.call_json("test", max_json_retries=1)
        assert result is None

    def test_call_json_appends_hint_to_system(self):
        llm = MockLLM(responses=['{"ok": 1}'])
        caller = LLMCaller(llm)
        caller.call_json("test", system="Custom system")
        # System should include custom + JSON hint
        assert "Custom system" in llm.last_system
        assert JSON_OUTPUT_HINT.strip() in llm.last_system

    def test_call_json_uses_default_constraints_when_no_system(self):
        llm = MockLLM(responses=['{"ok": 1}'])
        caller = LLMCaller(llm)
        caller.call_json("test")
        assert DEFAULT_SYSTEM_CONSTRAINTS in llm.last_system


# ===================== 缓存 =====================

class TestLLMCallerCache:

    def setup_method(self):
        LLMCaller._cache.clear()

    def test_cache_hit(self):
        llm = MockLLM(responses=["first", "second"])
        caller = LLMCaller(llm, cache_ttl=300)
        resp1 = caller.call("test")
        resp2 = caller.call("test")  # same message → cache hit
        assert resp1.content == resp2.content == "first"
        assert llm.call_count == 1  # only one actual API call

    def test_cache_miss_different_messages(self):
        llm = MockLLM(responses=["first", "second"])
        caller = LLMCaller(llm, cache_ttl=300)
        resp1 = caller.call("test1")
        resp2 = caller.call("test2")
        assert resp1.content == "first"
        assert resp2.content == "second"
        assert llm.call_count == 2

    def test_cache_disabled(self):
        llm = MockLLM(responses=["first", "second"])
        caller = LLMCaller(llm, cache_ttl=300)
        caller.call("test", use_cache=False)
        caller.call("test", use_cache=False)
        assert llm.call_count == 2

    def test_cache_expired(self):
        llm = MockLLM(responses=["first", "second"])
        caller = LLMCaller(llm, cache_ttl=1)
        caller.call("test")
        # Manually expire
        for key in list(caller._cache.keys()):
            caller._cache[key].timestamp -= 10
        caller.call("test")
        assert llm.call_count == 2

    def test_clear_cache(self):
        llm = MockLLM(responses=["first"])
        caller = LLMCaller(llm, cache_ttl=300)
        caller.call("test")
        count = caller.clear_cache()
        assert count == 1
        assert len(caller._cache) == 0


# ===================== 输入长度校验 =====================

class TestInputLengthValidation:

    def setup_method(self):
        LLMCaller._cache.clear()

    def test_long_input_logs_warning(self, caplog):
        llm = MockLLM(responses=["OK"])
        caller = LLMCaller(llm, max_input_chars=100)
        with caplog.at_level("WARNING"):
            caller.call("x" * 200, use_cache=False)
        assert any("超过阈值" in record.message for record in caplog.records)

    def test_normal_input_no_warning(self, caplog):
        llm = MockLLM(responses=["OK"])
        caller = LLMCaller(llm, max_input_chars=10000)
        with caplog.at_level("WARNING"):
            caller.call("short text", use_cache=False)
        assert not any("超过阈值" in r.message for r in caplog.records)


# ===================== Function Calling =====================

class TestCallWithTools:

    def setup_method(self):
        LLMCaller._cache.clear()

    def test_call_with_tools(self):
        llm = MockLLM()
        caller = LLMCaller(llm)
        resp = caller.call_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "tool1"}}],
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["function"]["name"] == "test_tool"


# ===================== ModelRouter 集成 =====================

class TestModelRouterIntegration:

    def test_get_caller(self):
        from financial_rag.llm.model_router import ModelRouter
        # ModelRouter needs an API key, mock it
        router = ModelRouter(api_key="test-key")
        caller = router.get_caller(task_type="analysis")
        assert isinstance(caller, LLMCaller)

    def test_get_caller_for_agent(self):
        from financial_rag.llm.model_router import ModelRouter
        router = ModelRouter(api_key="test-key")
        caller = router.get_caller_for_agent("ReportAgent")
        assert isinstance(caller, LLMCaller)


# ===================== 工厂函数 =====================

class TestGetCaller:

    def test_get_caller_factory(self):
        llm = MockLLM()
        caller = get_caller(llm, max_retries=5, cache_ttl=60)
        assert isinstance(caller, LLMCaller)
        assert caller.max_retries == 5
        assert caller.cache_ttl == 60
