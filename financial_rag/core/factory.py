"""
工厂函数 — create_orchestrator, create_hybrid_retriever, setup_environment

集中管理所有组件的创建逻辑，避免散落在各处。
"""
import os
import logging
from typing import Any

from financial_rag.core.base import ExecutionMode
from financial_rag.core.orchestrator import AgentOrchestrator, CoordinatorConfig

logger = logging.getLogger(__name__)


def create_orchestrator() -> AgentOrchestrator:
    """创建 3-Agent 链: Ingestion → Extraction → Report"""
    from financial_rag.agents.ingestion_agent import IngestionAgent
    from financial_rag.agents.extraction_agent import ExtractionAgent
    from financial_rag.agents.report_agent import ReportAgent

    orch = AgentOrchestrator(
        CoordinatorConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            verbose=True,
            max_retries=2,
        )
    )
    orch.register_all(
        IngestionAgent(),
        ExtractionAgent(),
        ReportAgent(),
    )
    return orch


def create_hybrid_retriever() -> Any:
    """创建带阿里 Embedding + Rerank + Jieba 分词的混合检索器"""
    from financial_rag.llm import get_embedding, get_reranker
    from financial_rag.retrievers import HybridRetriever, jieba_tokenizer

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    tokenizer = None
    try:
        tokenizer = jieba_tokenizer()
    except ImportError:
        pass

    if not api_key:
        print("[WARN] 未设置 DASHSCOPE_API_KEY，回退到纯本地检索")
        return HybridRetriever(tokenizer=tokenizer)

    return HybridRetriever(
        embedder=get_embedding(api_key=api_key),
        reranker=get_reranker(api_key=api_key),
        tokenizer=tokenizer,
    )


def setup_environment() -> bool:
    """初始化环境，返回是否有 API Key"""
    from financial_rag.config import config as _cfg

    for d in [_cfg.data_dir, _cfg.kb_dir, _cfg.output_dir]:
        os.makedirs(d, exist_ok=True)

    has_key = bool(_cfg.llm.api_key)
    if not has_key:
        print("[WARN] 未设置 DASHSCOPE_API_KEY，使用纯本地模式")
        print("       设置: export DASHSCOPE_API_KEY=sk-xxx")
        print("       获取: https://bailian.console.aliyun.com/\n")
    else:
        print(f"[INFO] DashScope API 已配置，模型: {_cfg.llm.model}")
    return has_key
