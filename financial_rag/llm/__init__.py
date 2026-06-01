"""
阿里百炼 DashScope LLM / Embedding / Rerank 客户端

模型清单:
- LLM:      qwen-plus / qwen-max / qwen-turbo / qwen3-235b-a22b
- Embedding: text-embedding-v3
- Rerank:    gte-rerank
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

__all__ = [
    "DashScopeLLM",
    "DashScopeEmbedding",
    "DashScopeReranker",
    "create_client",
    "get_llm",
    "get_embedding",
    "get_reranker",
]
