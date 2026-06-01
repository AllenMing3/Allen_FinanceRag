"""
阿里百炼 DashScope API 客户端

封装三个核心能力:
1. LLM Chat (Qwen 系列)
2. Text Embedding (text-embedding-v3)
3. Rerank (gte-rerank)

文档: https://help.aliyun.com/document_detail/2712195.html
"""
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 可选依赖 — 首次调用时自动安装提示
try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    dashscope = None  # type: ignore


# ===================== 数据类 =====================

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict = field(default_factory=dict)
    finish_reason: str = ""

@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    model: str
    dimensions: int = 1024
    usage: Dict = field(default_factory=dict)

@dataclass
class RerankResult:
    index: int
    score: float           # 0~1 相关度分数
    document: str
    relevance_level: str = ""  # high / medium / low


# ===================== LLM 客户端 =====================

class DashScopeLLM:
    """阿里百炼 Chat 模型客户端"""

    # 可用模型列表
    MODELS = {
        "qwen-turbo":        "qwen-turbo-latest",
        "qwen-plus":         "qwen-plus-latest",
        "qwen-max":          "qwen-max-latest",
        "qwen3-235b":        "qwen3-235b-a22b",
        "qwen-coder-plus":   "qwen-coder-plus-latest",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 0.9,
    ):
        self.api_key = api_key
        self.model = self.MODELS.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._check_sdk()

    def _check_sdk(self):
        if not HAS_DASHSCOPE:
            raise ImportError(
                "请安装 dashscope: pip install dashscope"
            )

    # ---------- 单轮 ----------

    def chat(
        self,
        messages: Union[str, List[Dict]],
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """单轮对话"""
        if isinstance(messages, str):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": messages})
        else:
            msgs = list(messages)

        resp = dashscope.Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=msgs,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            result_format="message",
        )

        if resp.status_code == 200:
            return LLMResponse(
                content=resp.output.choices[0].message.content,
                model=self.model,
                usage=resp.usage.__dict__ if resp.usage else {},
                finish_reason=resp.output.choices[0].finish_reason or "",
            )
        else:
            raise RuntimeError(
                f"DashScope LLM error: code={resp.status_code}, "
                f"message={resp.message}"
            )

    # ---------- 流式 ----------

    def chat_stream(
        self,
        messages: Union[str, List[Dict]],
        system: Optional[str] = None,
        **kwargs,
    ):
        """流式对话 — 返回生成器"""
        if isinstance(messages, str):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": messages})
        else:
            msgs = list(messages)

        resp = dashscope.Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=msgs,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            result_format="message",
            stream=True,
            incremental_output=True,
        )

        for chunk in resp:
            if chunk.status_code == 200:
                delta = chunk.output.choices[0].message.content
                finish = chunk.output.choices[0].finish_reason
                yield delta, finish
            else:
                raise RuntimeError(f"DashScope stream error: {chunk.message}")


# ===================== Embedding 客户端 =====================

class DashScopeEmbedding:
    """阿里百炼 Text Embedding 客户端

    支持模型: text-embedding-v3 (1024维 默认)
    批量处理，自动分批（单次最多 25 条）
    """

    BATCH_SIZE = 25

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._check_sdk()

    def _check_sdk(self):
        if not HAS_DASHSCOPE:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def embed(self, texts: Union[str, List[str]]) -> EmbeddingResponse:
        """文本转向量 — 自动分批"""
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else texts

        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(batch), self.BATCH_SIZE):
            chunk = batch[i:i + self.BATCH_SIZE]
            resp = dashscope.TextEmbedding.call(
                api_key=self.api_key,
                model=self.model,
                input=chunk,
            )

            if resp.status_code == 200:
                for emb in resp.output["embeddings"]:
                    all_embeddings.append(emb["embedding"])
                total_tokens += resp.usage.get("total_tokens", 0)
            else:
                raise RuntimeError(
                    f"DashScope Embedding error: code={resp.status_code}, "
                    f"message={resp.message}"
                )

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=self.model,
            dimensions=self.dimensions,
            usage={"total_tokens": total_tokens, "texts": len(batch)},
        )

    def embed_query(self, text: str) -> List[float]:
        """快捷方法: 单文本转向量"""
        return self.embed(text).embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """快捷方法: 批量文本转向量"""
        return self.embed(texts).embeddings


# ===================== Rerank 客户端 =====================

class DashScopeReranker:
    """阿里百炼 Rerank 客户端

    模型: gte-rerank
    对检索到的候选文档重排序，提升检索精度
    """

    BATCH_SIZE = 20  # 单次最大文档数

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gte-rerank",
        top_n: int = 5,
        return_documents: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self.return_documents = return_documents
        self._check_sdk()

    def _check_sdk(self):
        if not HAS_DASHSCOPE:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        对文档列表按与 query 的相关度重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_n: 返回前 N 条，默认用 init 的值

        Returns:
            按相关度降序排列的结果列表
        """
        if not documents:
            return []

        top_n = top_n or self.top_n
        all_results = []

        # 分批处理
        for i in range(0, len(documents), self.BATCH_SIZE):
            batch = documents[i:i + self.BATCH_SIZE]

            resp = dashscope.TextReRank.call(
                api_key=self.api_key,
                model=self.model,
                query=query,
                documents=batch,
                top_n=min(top_n, len(batch)),
                return_documents=self.return_documents,
                parameters={"return_scores": True},
            )

            if resp.status_code == 200:
                for r in resp.output.get("results", []):
                    all_results.append(RerankResult(
                        index=i + r["index"],
                        score=r.get("relevance_score", 0),
                        document=r.get("document", batch[r["index"]]),
                        relevance_level=self._level(r.get("relevance_score", 0)),
                    ))
            else:
                raise RuntimeError(
                    f"DashScope Rerank error: code={resp.status_code}, "
                    f"message={resp.message}"
                )

        # 全局排序取 top_n
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_n]

    # noinspection PyMethodMayBeStatic
    def _level(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        return "low"


# ===================== 便捷工厂 =====================

_client_cache: Dict[str, object] = {}


def create_client(
    client_type: str,           # "llm" | "embedding" | "reranker"
    api_key: Optional[str] = None,
    **kwargs,
):
    """创建 DashScope 客户端"""
    import os
    key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    cache_key = f"{client_type}:{kwargs}"

    if cache_key in _client_cache:
        return _client_cache[cache_key]

    if not key:
        raise ValueError(
            "请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key"
        )

    if client_type == "llm":
        client = DashScopeLLM(api_key=key, **kwargs)
    elif client_type == "embedding":
        client = DashScopeEmbedding(api_key=key, **kwargs)
    elif client_type == "reranker":
        client = DashScopeReranker(api_key=key, **kwargs)
    else:
        raise ValueError(f"未知客户端类型: {client_type}")

    _client_cache[cache_key] = client
    return client


def get_llm(**kwargs):
    """获取 LLM 单例"""
    return create_client("llm", **kwargs)


def get_embedding(**kwargs):
    """获取 Embedding 单例"""
    return create_client("embedding", **kwargs)


def get_reranker(**kwargs):
    """获取 Rerank 单例"""
    return create_client("reranker", **kwargs)
