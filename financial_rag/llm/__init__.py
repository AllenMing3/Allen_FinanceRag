"""
阿里百炼 DashScope LLM / Embedding / Rerank 客户端

模型清单:
- LLM:      qwen-plus / qwen-max / qwen-turbo / qwen3-235b-a22b
- Embedding: text-embedding-v3
- Rerank:    qwen3-rerank

智能路由:
- ModelRouter: 按任务复杂度自动选择模型

调用保护层:
- LLMCaller: 重试 + 缓存 + 结构化 JSON + 输入校验 + 防幻觉约束
"""
from .dashscope_client import (
    DashScopeLLM,
    DashScopeEmbedding,
    DashScopeReranker,
    create_client,
    get_llm,
    get_embedding,
    get_reranker,
)
from .caller import (
    LLMCaller,
    get_caller,
    parse_json_from_text,
    parse_json_list_from_text,
    DEFAULT_SYSTEM_CONSTRAINTS,
)
from .model_router import (
    ModelRouter,
    ModelTier,
    TaskComplexity,
    BudgetConfig,
    ModelRouterStats,
    TIER_MODEL_MAP,
    TASK_COMPLEXITY_MAP,
    COMPLEXITY_TO_TIER,
)

__all__ = [
    "DashScopeLLM",
    "DashScopeEmbedding",
    "DashScopeReranker",
    "create_client",
    "get_llm",
    "get_embedding",
    "get_reranker",
    "LLMCaller",
    "get_caller",
    "parse_json_from_text",
    "parse_json_list_from_text",
    "DEFAULT_SYSTEM_CONSTRAINTS",
    "ModelRouter",
    "ModelTier",
    "TaskComplexity",
    "BudgetConfig",
    "ModelRouterStats",
    "TIER_MODEL_MAP",
    "TASK_COMPLEXITY_MAP",
    "COMPLEXITY_TO_TIER",
]
