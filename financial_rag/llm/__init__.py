"""
阿里百炼 DashScope LLM / Embedding / Rerank 客户端

模型清单:
- LLM:      qwen-plus / qwen-max / qwen-turbo / qwen3-235b-a22b
- Embedding: text-embedding-v3
- Rerank:    qwen3-rerank

智能路由:
- ModelRouter: 按任务复杂度自动选择模型
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
    "ModelRouter",
    "ModelTier",
    "TaskComplexity",
    "BudgetConfig",
    "ModelRouterStats",
    "TIER_MODEL_MAP",
    "TASK_COMPLEXITY_MAP",
    "COMPLEXITY_TO_TIER",
]
