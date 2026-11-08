"""
financial_rag.observability.tracing — Phoenix / OpenTelemetry 埋点收口

横切关注点，绝不侵入业务逻辑。三条设计铁律：
- 默认关闭：is_enabled()==False 时 observe_span 透传 yield None，零开销、零副作用、不改返回值。
- 懒加载：openinference 属性名与 opentelemetry OTLP 导出器仅在启用时才 import；未装也不报错。
- 不阻断：init_phoenix 任何异常仅 logger.warning 并返回 False，绝不影响主流程与测试。

只装轻量 opentelemetry-sdk 也能埋点（属性名用字符串常量兜底，不强依赖 openinference 包）。
"""
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenInference 语义约定属性名
# 优先用官方 openinference.semconv 包，未装则用等价字符串常量兜底
# （两条路径产出的属性名完全一致，保证只装 otel-sdk 也能被 Phoenix 正确解析）
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 取决于是否安装 openinference
    from openinference.semconv.trace import SpanAttributes as _SA

    SPAN_KIND = _SA.OPENINFERENCE_SPAN_KIND
    INPUT_VALUE = _SA.INPUT_VALUE
    INPUT_MIME_TYPE = _SA.INPUT_MIME_TYPE
    OUTPUT_VALUE = _SA.OUTPUT_VALUE
    OUTPUT_MIME_TYPE = _SA.OUTPUT_MIME_TYPE
    LLM_MODEL_NAME = _SA.LLM_MODEL_NAME
    LLM_TOKEN_COUNT_PROMPT = _SA.LLM_TOKEN_COUNT_PROMPT
    LLM_TOKEN_COUNT_COMPLETION = _SA.LLM_TOKEN_COUNT_COMPLETION
    LLM_TOKEN_COUNT_TOTAL = _SA.LLM_TOKEN_COUNT_TOTAL
except (ImportError, AttributeError):  # pragma: no cover - 兜底：未装或属性名不符时回退字面量
    SPAN_KIND = "openinference.span.kind"
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"
    LLM_MODEL_NAME = "llm.model_name"
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"

# span kind 取值（OpenInferenceSpanKindValues 的字符串形式）
KIND_LLM = "LLM"
KIND_CHAIN = "CHAIN"
KIND_RETRIEVER = "RETRIEVER"
KIND_EMBEDDING = "EMBEDDING"
KIND_RERANKER = "RERANKER"
KIND_TOOL = "TOOL"

# 检索命中文档的属性名模板（OpenInference 用索引展开）
_DOC_CONTENT_TMPL = "retrieval.documents.{i}.document.content"
_DOC_SCORE_TMPL = "retrieval.documents.{i}.document.score"
_DOC_ID_TMPL = "retrieval.documents.{i}.document.id"


# ---------------------------------------------------------------------------
# 开关状态（缓存 config.observability.enable，避免每次调用重复读配置）
# ---------------------------------------------------------------------------
_enabled_cache: Optional[bool] = None


def is_enabled() -> bool:
    """是否启用 Phoenix 埋点（读 config.observability.enable，缓存结果）

    任何异常都降级为 False，确保埋点开关永不成为主流程故障源。
    """
    global _enabled_cache
    if _enabled_cache is None:
        try:
            from financial_rag.config import config

            _enabled_cache = bool(config.observability.enable)
        except Exception as exc:  # pragma: no cover - 配置异常兜底
            logger.warning(f"[observability] 读取开关失败，按关闭处理: {exc}")
            _enabled_cache = False
    return _enabled_cache


def _reset_cache() -> None:
    """清除 is_enabled 缓存（供测试临时改开关后刷新）"""
    global _enabled_cache
    _enabled_cache = None


# ---------------------------------------------------------------------------
# OTLP 导出初始化（推送到外部独立运行的 Phoenix collector）
# ---------------------------------------------------------------------------
_phoenix_initialized = False


def init_phoenix() -> bool:
    """配置 OTLP/HTTP 导出器，把 span 推送到外部独立运行的 Phoenix collector

    主环境不依赖 phoenix 包（Phoenix 隔离在独立 .venv-obs）；仅需 opentelemetry-sdk
    与 opentelemetry-exporter-otlp-proto-http（见 pyproject.toml 的 observability extra）。
    懒加载 exporter：未装则告警并返回 False，绝不阻断主流程。成功返回 True（幂等）。
    """
    global _phoenix_initialized
    if _phoenix_initialized:
        return True

    # 1) 懒加载 OTLP/HTTP 导出器（未装 observability extra 时优雅降级）
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "[observability] 缺少 OTLP/HTTP 导出器，跳过可视化初始化"
            "（如需启用：uv sync --extra observability，并运行 scripts/start_phoenix.*）"
        )
        return False

    # 2) 配置 provider + BatchSpanProcessor，把 span 推到外部 Phoenix 的 /v1/traces
    try:
        from financial_rag.config import config

        obs = config.observability
        endpoint = obs.endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"

        # 复用已存在的全局 provider（如测试注入的 InMemorySpanExporter），否则新建
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider(
                resource=Resource.create({
                    "service.name": obs.project_name,
                    # Phoenix v21+ 用 resource 属性归属项目，不再认 x-phoenix-project-name
                    # header；缺这行所有 span 都会落进 default 项目（已实测验证）。
                    "openinference.project.name": obs.project_name,
                })
            )
            trace.set_tracer_provider(provider)

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"x-phoenix-project-name": obs.project_name},
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _phoenix_initialized = True
        logger.info(
            f"[observability] OTLP 导出已配置: project={obs.project_name}, "
            f"endpoint={endpoint}"
        )
        return True
    except Exception as exc:
        logger.warning(f"[observability] OTLP 导出配置失败，主流程不受影响: {exc}")
        return False


# ---------------------------------------------------------------------------
# 埋点上下文管理器
# ---------------------------------------------------------------------------
def set_attrs(span: Any, attributes: Optional[Dict[str, Any]]) -> None:
    """在 span 上安全批量写属性；span 为 None（未启用）或写入异常时静默 no-op

    让调用点无需自行判空/兜底，保持业务代码整洁。
    """
    if span is None or not attributes:
        return
    for key, value in attributes.items():
        try:
            span.set_attribute(key, value)
        except Exception as exc:  # pragma: no cover - 属性写入异常兜底
            logger.warning(f"[observability] 写属性失败 {key}: {exc}")


def _doc_field(doc: Any, *names: str) -> Any:
    """从 dict 或对象上按候选名取第一个非 None 值。"""
    if isinstance(doc, dict):
        for n in names:
            if doc.get(n) is not None:
                return doc[n]
        return None
    for n in names:
        v = getattr(doc, n, None)
        if v is not None:
            return v
    return None


def retrieval_doc_attrs(docs) -> Dict[str, Any]:
    """把检索命中文档列表转成 OpenInference 索引属性字典

    docs: 可迭代对象，每项可为 dict（检索返回）或对象（.text/.content）；
    兼容 text/content 正文、score/rerank_score 得分、id/doc_id 标识。
    content 截断到 500 字符，避免 span 过大。
    """
    attrs: Dict[str, Any] = {}
    for i, doc in enumerate(docs or []):
        text = _doc_field(doc, "text", "content") or ""
        attrs[_DOC_CONTENT_TMPL.format(i=i)] = str(text)[:500]
        score = _doc_field(doc, "score", "rerank_score")
        if score is not None:
            try:
                attrs[_DOC_SCORE_TMPL.format(i=i)] = float(score)
            except (TypeError, ValueError):
                pass
        doc_id = _doc_field(doc, "id", "doc_id")
        if doc_id is not None:
            attrs[_DOC_ID_TMPL.format(i=i)] = str(doc_id)
    return attrs


@contextmanager
def observe_span(
    name: str,
    kind: str = KIND_CHAIN,
    attributes: Optional[Dict[str, Any]] = None,
):
    """埋点上下文管理器：未启用时零开销透传，启用时开一个 OpenInference span

    用法::

        with observe_span("llm.call", KIND_LLM, {INPUT_VALUE: ...}) as span:
            resp = do_call()
            set_attrs(span, {OUTPUT_VALUE: resp.content})

    - 未启用（is_enabled()==False）或 OpenTelemetry 不可用时：yield None，
      被包裹逻辑照常执行，返回值不变、不产生 span、不抛异常。
    - 启用时：yield 真实 span，供调用方 set_attrs 补充 input/output/token 等属性。
    """
    if not is_enabled():
        yield None
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("financial_rag")
    except Exception as exc:
        logger.warning(f"[observability] OpenTelemetry 不可用，跳过 span '{name}': {exc}")
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        set_attrs(span, {SPAN_KIND: kind})
        set_attrs(span, attributes)
        yield span
