"""
Financial RAG 配置 — 与业务完全脱钩，默认使用阿里百炼 DashScope API
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 确保从任意 CWD 运行都能找到项目根目录的 .env
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_project_root / ".env", override=True)


@dataclass
class LLMConfig:
    """LLM 配置 — 阿里百炼 DashScope"""
    # 阿里百炼 API（默认）
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    provider: str = "dashscope"                                   # dashscope | openai（兼容）

    # 模型选择
    model: str = "qwen-plus"                                      # qwen-turbo / qwen-plus / qwen-max
    embedding_model: str = "text-embedding-v3"                    # 1024 维
    rerank_model: str = "gte-rerank"                              # 重排序模型

    # 生成参数
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 0.9


@dataclass
class RAGConfig:
    """RAG 配置"""
    vector_store_path: str = "./storage/financial_vector_store"
    similarity_top_k: int = 5
    response_mode: str = "compact"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class CoordinatorConfig:
    """Coordinate 架构配置"""
    execution_mode: str = "sequential"      # sequential / parallel / conditional
    max_parallel_agents: int = 3
    max_retries: int = 2
    timeout_seconds: float = 300.0
    verbose: bool = True


@dataclass
class PipelineConfig:
    """Indexer 架构配置"""
    hybrid_top_k: int = 10
    rrf_k: int = 60
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    min_faithfulness: float = 0.7
    min_source_count: int = 2


@dataclass
class ReflectionConfig:
    """Reflection 架构配置"""
    max_retrievals: int = 3
    max_steps: int = 6
    min_confidence: float = 0.6
    enable_self_reflection: bool = True
    hallucination_threshold: float = 0.6


@dataclass
class AppConfig:
    """应用总配置"""
    # 路径
    data_dir: str = "./data/financial"
    kb_dir: str = "./data/knowledge_base"
    output_dir: str = "./output"

    # 子配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)

    def __post_init__(self):
        for d in [self.data_dir, self.kb_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)


# 全局配置
config = AppConfig()
