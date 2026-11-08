"""
Test 可观测性层 — Phoenix/OpenTelemetry 埋点收口质量验证。

对应生产代码：
  - financial_rag/observability/tracing.py（is_enabled / init_phoenix / observe_span / set_attrs）
  - financial_rag/observability/__init__.py（导出面）

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定

核心契约（横切关注点铁律）：
  - no-op 安全：未启用时 observe_span 透传 yield None，被包裹逻辑返回值不变、不抛异常、不产生 span。
  - 离线可观测：用 InMemorySpanExporter 临时配 SDK provider，手动启用后跑埋点，
    断言 span 名/kind/input.value/output.value/token 属性正确 —— 不依赖真实 Phoenix、不联网。
  - 懒加载回退：未装 OTLP/HTTP 导出器时 init_phoenix() 返回 False 且不抛异常。

离线约束：conftest 已 setdefault PHOENIX_ENABLED=false；启用路径由 enable_tracing
fixture 临时改 config 并刷新缓存，用例结束恢复，绝不污染其他模块。
"""
import pytest

from financial_rag.config import config
from financial_rag.observability import (
    init_phoenix,
    is_enabled,
    observe_span,
    retrieval_doc_attrs,
    set_attrs,
)
from financial_rag.observability import tracing
from financial_rag.observability.tracing import (
    INPUT_VALUE,
    KIND_CHAIN,
    KIND_LLM,
    KIND_RETRIEVER,
    LLM_MODEL_NAME,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    OUTPUT_VALUE,
    SPAN_KIND,
)


# ===================== Fixtures =====================

@pytest.fixture(scope="module")
def span_exporter():
    """模块级：配置一次全局 SDK TracerProvider + InMemorySpanExporter。

    observe_span 走 trace.get_tracer() 读全局 provider；OTel 只允许 set 一次，
    故模块级建一个 exporter，各用例用 clear() 隔离。仅本模块会产出 span
    （其他模块 is_enabled()==False，observe_span 直接透传，不碰 tracer）。
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    exporter.clear()
    yield exporter
    exporter.clear()


@pytest.fixture
def enable_tracing():
    """临时启用埋点：改 config.observability.enable + 刷新缓存，用例结束恢复。"""
    old = config.observability.enable
    config.observability.enable = True
    tracing._reset_cache()
    try:
        yield
    finally:
        config.observability.enable = old
        tracing._reset_cache()


class _FakeDoc:
    """检索命中文档的最小替身（供 retrieval_doc_attrs 读取 text/score/id）。"""

    def __init__(self, text, score=None, id=None):
        self.text = text
        self.score = score
        self.id = id


# ===================== ① no-op 安全（默认关闭） =====================

class TestNoopSafety:
    def test_is_enabled_false_by_default(self):
        """契约：conftest 强制 PHOENIX_ENABLED=false，默认不启用埋点。"""
        tracing._reset_cache()
        assert is_enabled() is False

    def test_noop_yields_none_and_preserves_return(self, span_exporter):
        """证明效果：未启用时 observe_span 透传 yield None，被包裹函数返回值不变、不抛异常。"""
        assert is_enabled() is False
        span_exporter.clear()

        def compute():
            with observe_span("llm.call", KIND_LLM, {INPUT_VALUE: "q"}) as span:
                assert span is None  # 未启用 → 无真实 span
                return 42

        assert compute() == 42

    def test_noop_produces_no_span(self, span_exporter):
        """回归锁定：未启用时绝不产生任何 span（零副作用）。"""
        span_exporter.clear()
        with observe_span("retriever.search", KIND_RETRIEVER) as span:
            assert span is None
        assert span_exporter.get_finished_spans() == ()

    def test_set_attrs_noop_on_none_span(self):
        """安全：set_attrs 收到 None span 静默 no-op，不抛异常。"""
        # 不应抛异常
        set_attrs(None, {INPUT_VALUE: "x", OUTPUT_VALUE: "y"})
        set_attrs(None, None)
        set_attrs(None, {})


# ===================== ② 可观测（离线 InMemorySpanExporter 断言） =====================

class TestEnabledSpanCapture:
    def test_llm_span_attributes(self, span_exporter, enable_tracing):
        """可观测：启用后 LLM span 记录 name/kind/input/output/model/token 属性。"""
        assert is_enabled() is True
        span_exporter.clear()

        with observe_span("llm.call", KIND_LLM, {INPUT_VALUE: "hello"}) as span:
            assert span is not None
            set_attrs(span, {
                OUTPUT_VALUE: "world",
                LLM_MODEL_NAME: "qwen-plus",
                LLM_TOKEN_COUNT_PROMPT: 10,
                LLM_TOKEN_COUNT_COMPLETION: 5,
            })

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert spans[0].name == "llm.call"
        assert attrs[SPAN_KIND] == "LLM"
        assert attrs[INPUT_VALUE] == "hello"
        assert attrs[OUTPUT_VALUE] == "world"
        assert attrs[LLM_MODEL_NAME] == "qwen-plus"
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 10
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 5

    def test_retriever_span_with_doc_attrs(self, span_exporter, enable_tracing):
        """可观测：检索 span 记录 query 与命中文档索引属性（content/score）。"""
        span_exporter.clear()
        docs = [_FakeDoc("商汤2024营收50.3亿", score=0.91, id="d1"),
                _FakeDoc("生成式AI占比60%", score=0.85, id="d2")]

        with observe_span("retriever.search", KIND_RETRIEVER, {INPUT_VALUE: "商汤营收"}) as span:
            set_attrs(span, retrieval_doc_attrs(docs))

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs[SPAN_KIND] == "RETRIEVER"
        assert attrs[INPUT_VALUE] == "商汤营收"
        assert attrs["retrieval.documents.0.document.content"] == "商汤2024营收50.3亿"
        assert attrs["retrieval.documents.0.document.score"] == 0.91
        assert attrs["retrieval.documents.1.document.id"] == "d2"

    def test_nested_spans_parent_child(self, span_exporter, enable_tracing):
        """可观测：嵌套 observe_span 形成父子链路（上下文正确传播）。"""
        span_exporter.clear()
        with observe_span("outer.chain", KIND_CHAIN):
            with observe_span("inner.llm", KIND_LLM):
                pass

        by_name = {s.name: s for s in span_exporter.get_finished_spans()}
        assert set(by_name) == {"outer.chain", "inner.llm"}
        assert by_name["inner.llm"].parent.span_id == by_name["outer.chain"].context.span_id


# ===================== ③ 一致性 =====================

class TestConsistency:
    def test_same_input_deterministic_attrs(self, span_exporter, enable_tracing):
        """一致性：同输入多次埋点，产出的 span 属性完全确定、可复现。"""
        span_exporter.clear()
        for _ in range(3):
            with observe_span("llm.call", KIND_LLM, {INPUT_VALUE: "q"}) as span:
                set_attrs(span, {OUTPUT_VALUE: "a", LLM_MODEL_NAME: "qwen-plus"})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 3
        attr_dicts = [dict(s.attributes) for s in spans]
        assert attr_dicts[0] == attr_dicts[1] == attr_dicts[2]


# ===================== ④ 懒加载回退 + retrieval_doc_attrs 契约 =====================

class TestLazyLoadFallback:
    def test_init_phoenix_false_when_exporter_missing(self, monkeypatch):
        """回归锁定：模拟未装 OTLP/HTTP 导出器，init_phoenix 返回 False 且不抛异常。"""
        import sys

        monkeypatch.setattr(tracing, "_phoenix_initialized", False)
        # sys.modules[name]=None 会让 `from name import ...` 抛 ImportError，
        # 以此模拟主环境未装 opentelemetry-exporter-otlp-proto-http（未 --extra observability）
        monkeypatch.setitem(
            sys.modules,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            None,
        )
        # 不应抛异常，返回 False，且不置初始化标记
        assert tracing.init_phoenix() is False
        assert tracing._phoenix_initialized is False

    def test_init_phoenix_idempotent_when_initialized(self, monkeypatch):
        """一致性：已初始化标记为 True 时，init_phoenix 幂等直接返回 True，不重复拉起。"""
        monkeypatch.setattr(tracing, "_phoenix_initialized", True)
        assert tracing.init_phoenix() is True


class TestRetrievalDocAttrs:
    def test_empty_docs_yield_empty_attrs(self):
        """边界：空文档列表 → 空属性字典，不抛异常。"""
        assert retrieval_doc_attrs([]) == {}
        assert retrieval_doc_attrs(None) == {}

    def test_content_truncated_to_500(self):
        """契约：content 截断到 500 字符，避免 span 过大。"""
        long_text = "营" * 800
        attrs = retrieval_doc_attrs([_FakeDoc(long_text, score=0.5)])
        assert len(attrs["retrieval.documents.0.document.content"]) == 500

    def test_missing_score_omitted(self):
        """契约：无 score 的文档不写 score 属性（避免脏值）。"""
        attrs = retrieval_doc_attrs([_FakeDoc("文本", score=None)])
        assert "retrieval.documents.0.document.content" in attrs
        assert "retrieval.documents.0.document.score" not in attrs

    def test_content_fallback_to_content_attr(self):
        """兼容：文档无 .text 时回退读 .content 属性。"""
        class Doc:
            content = "回退内容"
        attrs = retrieval_doc_attrs([Doc()])
        assert attrs["retrieval.documents.0.document.content"] == "回退内容"
