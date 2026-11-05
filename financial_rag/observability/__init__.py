"""
financial_rag.observability — 可观测性层（Arize Phoenix / OpenTelemetry）

横切能力域，与 core/ guard/ llm/ 平级。职责：把"查询 → 检索 → 生成"主干
的每一步以 OpenInference span 推送到 Phoenix UI，可视化 prompt/response/token/latency。

设计铁律：默认关闭、懒加载重依赖、任何异常都不阻断主流程（详见 tracing.py）。

用法:
    from financial_rag.observability import init_phoenix, is_enabled, observe_span
    from financial_rag.observability.tracing import KIND_LLM, INPUT_VALUE, set_attrs
"""
from .tracing import (
    init_phoenix,
    is_enabled,
    observe_span,
    set_attrs,
    retrieval_doc_attrs,
    KIND_LLM,
    KIND_CHAIN,
    KIND_RETRIEVER,
    KIND_EMBEDDING,
    KIND_RERANKER,
    KIND_TOOL,
    SPAN_KIND,
    INPUT_VALUE,
    INPUT_MIME_TYPE,
    OUTPUT_VALUE,
    OUTPUT_MIME_TYPE,
    LLM_MODEL_NAME,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_TOTAL,
)

__all__ = [
    "init_phoenix",
    "is_enabled",
    "observe_span",
    "set_attrs",
    "retrieval_doc_attrs",
    "KIND_LLM",
    "KIND_CHAIN",
    "KIND_RETRIEVER",
    "KIND_EMBEDDING",
    "KIND_RERANKER",
    "KIND_TOOL",
    "SPAN_KIND",
    "INPUT_VALUE",
    "INPUT_MIME_TYPE",
    "OUTPUT_VALUE",
    "OUTPUT_MIME_TYPE",
    "LLM_MODEL_NAME",
    "LLM_TOKEN_COUNT_PROMPT",
    "LLM_TOKEN_COUNT_COMPLETION",
    "LLM_TOKEN_COUNT_TOTAL",
]
