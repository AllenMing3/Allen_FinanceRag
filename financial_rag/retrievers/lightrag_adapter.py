"""
LightRAG Adapter — 知识图谱适配器

将 LightRAG async SDK 封装为同步接口，供 Pipeline 和 Ingest 使用。

设计原则:
    1. 与 experiments/lightrag_experiment.py 解耦 — 实验脚本保持独立
    2. async → sync 桥接 — 内部维护 event loop，对外暴露同步 API
    3. SimpleTokenizer — 避免 tiktoken 下载被墙
    4. 返回格式与 retrieved_items 对齐 — 图谱结果可直接注入 AgentContext

存储:
    LightRAG 内部用 JSON + GraphML 文件，放在 working_dir 下。
    无需外部数据库。

用法:
    adapter = LightRAGAdapter(working_dir="./data/knowledge_base/lightrag", api_key="sk-...")
    adapter.initialize()
    adapter.insert_texts(["商汤科技2024年营收50.2亿..."], [{"source": "annual_report.pdf"}])
    results = adapter.query("商汤科技营收增长了多少？")
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ===================== SimpleTokenizer (避免 tiktoken 下载) =====================

class _SimpleInner:
    """Inner tokenizer — 用字符索引作为 token，decode 用索引还原文本"""
    _last_text: str = ""

    def encode(self, content, **kwargs):
        self._last_text = content
        return list(range(len(content)))

    def decode(self, tokens):
        if not tokens or not self._last_text:
            return ""
        start = min(tokens)
        end = max(tokens) + 1
        return self._last_text[start:end]


def _create_simple_tokenizer():
    """创建 SimpleTokenizer，避免 tiktoken 依赖"""
    from lightrag.utils import Tokenizer
    return Tokenizer(model_name="custom", tokenizer=_SimpleInner())


# ===================== LLM / Embedding 适配函数工厂 =====================

def _make_llm_func(api_key: str, model: str = "qwen-plus"):
    """创建 LLM 适配函数 — dashscope SDK 直调"""

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        import dashscope
        from dashscope import Generation

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})

        # 移除 LightRAG 传入的不兼容参数
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("token_tracker", None)

        result_format = "message"
        if kwargs.pop("response_format", None):
            result_format = "json_object"

        resp = Generation.call(
            model=model,
            messages=messages,
            api_key=api_key,
            result_format=result_format,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
        )

        if resp.status_code != 200:
            raise RuntimeError(f"DashScope LLM error: {resp.code} - {resp.message}")

        return resp.output.choices[0].message.content

    return llm_func


def _make_embedding_func(api_key: str, model: str = "text-embedding-v3", dimension: int = 1024):
    """创建 Embedding 适配函数 — dashscope SDK 直调"""

    async def embedding_func(texts):
        import dashscope
        from dashscope import TextEmbedding
        import numpy as np

        resp = TextEmbedding.call(
            model=model,
            input=texts,
            api_key=api_key,
            dimension=dimension,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"DashScope Embedding error: {resp.code} - {resp.message}")

        embeddings = [item["embedding"] for item in resp.output["embeddings"]]
        return np.array(embeddings, dtype=np.float32)

    return embedding_func


# ===================== LightRAG Adapter =====================

class LightRAGAdapter:
    """
    LightRAG 知识图谱适配器 — 封装 async SDK 为同步接口

    仅接收 PDF/图片解析后的文本，通过实体-关系抽取构建知识图谱。
    查询时返回与 retrieved_items 对齐的格式，供 Pipeline 融合。
    """

    def __init__(
        self,
        working_dir: str = "./data/knowledge_base/lightrag",
        api_key: str = "",
        llm_model: str = "qwen-plus",
        embedding_model: str = "text-embedding-v3",
        embedding_dim: int = 1024,
        chunk_token_size: int = 300,
        chunk_overlap_token_size: int = 50,
        entity_extract_max_gleaning: int = 1,
    ):
        self.working_dir = working_dir
        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        self.entity_extract_max_gleaning = entity_extract_max_gleaning

        self._rag = None  # LightRAG instance
        self._initialized = False

    # ---------- lifecycle ----------

    def initialize(self):
        """初始化 LightRAG 实例 + 加载已有图谱存储"""
        if self._initialized:
            return

        if not self.api_key:
            logger.warning("[LightRAG] API key 未设置，图谱功能不可用")
            return

        os.makedirs(self.working_dir, exist_ok=True)

        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        # 抑制 lightrag 库内部噪音日志（KV load、Role LLM 等）
        import logging as _logging
        _logging.getLogger("lightrag").setLevel(_logging.WARNING)
        _logging.getLogger("nano-vectordb").setLevel(_logging.WARNING)

        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=_make_llm_func(self.api_key, self.llm_model),
            llm_model_name=self.llm_model,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                func=_make_embedding_func(self.api_key, self.embedding_model, self.embedding_dim),
                max_token_size=8000,
            ),
            embedding_batch_num=5,
            entity_extract_max_gleaning=self.entity_extract_max_gleaning,
            chunk_token_size=self.chunk_token_size,
            chunk_overlap_token_size=self.chunk_overlap_token_size,
            tokenizer=_create_simple_tokenizer(),
        )

        # 初始化存储（加载已有图谱数据）
        self._run_async(self._rag.initialize_storages())
        self._initialized = True
        logger.debug(f"[LightRAG] 初始化完成")

    def finalize(self):
        """关闭并持久化存储"""
        if self._rag and self._initialized:
            try:
                self._run_async(self._rag.finalize_storages())
                logger.info("[LightRAG] 存储已持久化")
            except Exception as e:
                logger.warning(f"[LightRAG] finalize 失败: {e}")
            self._initialized = False

    # ---------- insert ----------

    def insert_texts(self, texts: List[str], source_meta: Optional[List[Dict]] = None):
        """
        将 PDF/图片解析的文本插入图谱（触发实体-关系抽取）。

        Args:
            texts: 文本列表
            source_meta: 对应的元数据列表（可选，用于日志追踪）
        """
        if not self._initialized or not self._rag:
            logger.warning("[LightRAG] 未初始化，跳过插入")
            return

        if not texts:
            return

        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return

        source_info = ""
        if source_meta:
            sources = [m.get("source", m.get("original_file", "?")) for m in source_meta]
            source_info = f" (来源: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''})"

        logger.info(f"[LightRAG] 插入 {len(valid_texts)} 段文本到图谱{source_info}...")

        try:
            self._run_async(self._rag.ainsert(valid_texts))
            logger.info(f"[LightRAG] 插入完成")
        except Exception as e:
            logger.error(f"[LightRAG] 插入失败: {e}")

    # ---------- query ----------

    def query(
        self,
        text: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> List[Dict]:
        """
        查询知识图谱，返回与 retrieved_items 对齐的格式。

        Args:
            text: 查询文本
            mode: 查询模式 (local / global / hybrid / mix)
            top_k: 返回条数（LightRAG 内部不直接支持 top_k，这里用于截断）

        Returns:
            List[Dict]，每个 Dict 格式:
            {"text": "...", "meta": {"_source": "graph", "graph_mode": "hybrid", ...}}
        """
        if not self._initialized or not self._rag:
            return []

        if not text or not text.strip():
            return []

        try:
            from lightrag import QueryParam

            result = self._run_async(
                self._rag.aquery(text, param=QueryParam(mode=mode))
            )

            if not result:
                return []

            # LightRAG 返回的是一个字符串（综合回答）
            # 将其包装为与 retrieved_items 对齐的格式
            result_text = result if isinstance(result, str) else str(result)

            if not result_text.strip():
                return []

            return [{
                "text": result_text.strip(),
                "meta": {
                    "_source": "graph",
                    "graph_mode": mode,
                    "doc_type": "图谱查询",
                },
            }]

        except Exception as e:
            logger.warning(f"[LightRAG] 查询失败 ({mode}): {e}")
            return []

    # ---------- inspect ----------

    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not self._initialized or not self._rag:
            return {"initialized": False}

        stats: Dict[str, Any] = {"initialized": True, "working_dir": self.working_dir}

        try:
            labels = self._run_async(self._rag.get_graph_labels())
            stats["entity_count"] = len(labels) if labels else 0
            stats["sample_labels"] = (labels[:10] if labels else [])
        except Exception as e:
            stats["labels_error"] = str(e)

        return stats

    # ---------- async bridge ----------

    def _run_async(self, coro):
        """async → sync 桥接

        调用场景（ingest_router sync handler / pipeline / graph_tools）
        均运行在无 running event loop 的线程中，直接 asyncio.run 即可。
        """
        return asyncio.run(coro)
