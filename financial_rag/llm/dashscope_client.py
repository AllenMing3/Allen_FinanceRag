"""
阿里百炼 DashScope API 客户端

封装三个核心能力:
1. LLM Chat (Qwen 系列)
2. Text Embedding (text-embedding-v3)
3. Rerank (qwen3-rerank)

文档: https://help.aliyun.com/document_detail/2712195.html
"""
import os
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

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
    tool_calls: List[Dict] = field(default_factory=list)  # function calling 返回的工具调用

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

    # ---------- Function Calling ----------

    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: str = "auto",
        **kwargs,
    ) -> LLMResponse:
        """Function Calling 模式 — 发送 tools 定义，解析 LLM 返回的工具调用。

        DashScope Qwen 系列支持 OpenAI 兼容的 tools/tool_choice 参数。

        Args:
            messages: 对话历史
            tools: 工具定义列表 (OpenAI JSON Schema 格式)
            tool_choice: "auto" | "required" | "none" | {"type": "function", "function": {"name": "xxx"}}
            **kwargs: 覆盖 temperature/max_tokens 等

        Returns:
            LLMResponse，其中 tool_calls 包含 LLM 选择的工具调用
        """
        resp = dashscope.Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            result_format="message",
        )

        if resp.status_code == 200:
            choice = resp.output.choices[0]
            msg = choice.message
            # 提取 tool_calls（可能为 None）
            tool_calls_raw = getattr(msg, "tool_calls", None) or []
            # 标准化格式
            tool_calls = []
            for tc in tool_calls_raw:
                if hasattr(tc, '__dict__'):
                    tc = tc.__dict__
                tc = dict(tc)
                if "function" in tc and hasattr(tc["function"], '__dict__'):
                    tc["function"] = tc["function"].__dict__
                tool_calls.append(tc)

            return LLMResponse(
                content=msg.content or "",
                model=self.model,
                usage=resp.usage.__dict__ if resp.usage else {},
                finish_reason=choice.finish_reason or "",
                tool_calls=tool_calls,
            )
        else:
            raise RuntimeError(
                f"DashScope LLM error: code={resp.status_code}, "
                f"message={resp.message}"
            )

    # ---------- 多模态 (图片理解) ----------

    def describe_image(
        self,
        image_path: str,
        prompt: str = "请详细描述这张图片的所有可见内容。",
        model: str = "qwen-vl-plus",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """用多模态模型理解图片内容（传输层，不含业务 prompt）。

        业务层 prompt 由 tools/document_parse_tools.py 组装并传入。

        Args:
            image_path: 本地图片文件的绝对路径
            prompt: 发给多模态模型的用户提示词（由上层 tool 组装）
            model: 多模态模型，可选 qwen-vl-plus / qwen-vl-max
            max_tokens: 最大输出 token 数

        Returns:
            LLMResponse，content 为模型返回的文本
        """
        import os
        abs_path = os.path.abspath(image_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"图片文件不存在: {abs_path}")

        messages = [
            {"role": "user", "content": [
                {"image": f"file://{abs_path}"},
                {"text": prompt},
            ]}
        ]

        resp = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )

        if resp.status_code == 200:
            content = resp.output.choices[0].message.content
            # qwen-vl 返回的 content 可能是 list 格式
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                content = "\n".join(text_parts)
            return LLMResponse(
                content=content,
                model=model,
                usage=resp.usage.__dict__ if resp.usage else {},
                finish_reason=resp.output.choices[0].finish_reason or "",
            )
        else:
            raise RuntimeError(
                f"DashScope MultiModal error: code={resp.status_code}, "
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
    批量处理，自动分批（单次最多 10 条），多批并发（ThreadPoolExecutor）
    """

    BATCH_SIZE = 10       # DashScope 单次 API 上限 10 条
    MAX_WORKERS = 5       # 并发线程数（控制 QPS，避免 429）

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

    def _call_batch(self, chunk: List[str]) -> tuple:
        """单批 API 调用（线程内执行）

        Returns:
            (embeddings_list, total_tokens)
        """
        resp = dashscope.TextEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=chunk,
        )
        if resp.status_code == 200:
            embs = [item["embedding"] for item in resp.output["embeddings"]]
            tokens = resp.usage.get("total_tokens", 0)
            return embs, tokens
        else:
            raise RuntimeError(
                f"DashScope Embedding error: code={resp.status_code}, "
                f"message={resp.message}"
            )

    def embed(self, texts: Union[str, List[str]]) -> EmbeddingResponse:
        """文本转向量 — 自动分批 + 并发执行

        单批(≤10条)直接调用；多批时 ThreadPoolExecutor 并发，
        200 条文本: 20 批 × 5 并发 = 4 轮，耗时从 ~20s 降至 ~4s。
        """
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else texts

        # 切分为 API 批次
        chunks = [batch[i:i + self.BATCH_SIZE]
                  for i in range(0, len(batch), self.BATCH_SIZE)]

        if not chunks:
            return EmbeddingResponse(
                embeddings=[], model=self.model,
                dimensions=self.dimensions, usage={"total_tokens": 0, "texts": 0},
            )

        # 单批直接调用（无需线程池开销）
        if len(chunks) == 1:
            embs, tokens = self._call_batch(chunks[0])
            return EmbeddingResponse(
                embeddings=embs, model=self.model,
                dimensions=self.dimensions,
                usage={"total_tokens": tokens, "texts": len(batch)},
            )

        # 多批并发（pool.map 保证结果顺序与输入一致）
        all_embeddings: List[List[float]] = []
        total_tokens = 0

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            for embs, tokens in pool.map(self._call_batch, chunks):
                all_embeddings.extend(embs)
                total_tokens += tokens

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

    模型: qwen3-rerank
    对检索到的候选文档重排序，提升检索精度
    """

    BATCH_SIZE = 20  # 单次最大文档数

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen3-rerank",
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
    # Bug fix: 用 is None 而非 or，避免空字符串被错误回退
    key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
    # Bug fix: 把 api_key 放入 cache key，防止不同 key 返回缓存的旧客户端
    cache_key = f"{client_type}:{key}:{kwargs}"

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
