"""
向量检索引擎 — ChromaDB 语义检索 + Jaccard 回退

职责:
- ChromaDB ANN 语义检索（有 embedder 时）
- Jaccard 回退（无 API 时）
- 持久化向量存储（Chroma PersistentClient）
"""
import hashlib
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Chroma metadata 只支持 flat dict (str/int/float/bool)
_CHROMA_META_SKIP = {"_classify", "_rejected", "chunk_start"}


def _flatten_meta(meta: Dict) -> Dict:
    """将 meta dict 扁平化为 Chroma 兼容格式 (只保留 str/int/float/bool)"""
    flat = {}
    for k, v in meta.items():
        if k in _CHROMA_META_SKIP:
            continue
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif isinstance(v, dict):
            # 嵌套 dict → 跳过 (Chroma 不支持)
            continue
        elif isinstance(v, list):
            # list → 转为字符串
            flat[k] = str(v)
        else:
            flat[k] = str(v)
    return flat


class VectorEngine:
    """向量检索引擎 — ChromaDB + Jaccard fallback"""

    COLLECTION_NAME = "financial_rag_docs"

    def __init__(self, embedder: Any = None, tokenizer=None,
                 chroma_persist_dir: Optional[str] = None,
                 embedding_cache: Any = None):
        """
        Args:
            embedder: DashScopeEmbedding 实例
            tokenizer: 分词函数（Jaccard 回退用）
            chroma_persist_dir: Chroma 持久化目录，None 则用内存模式
            embedding_cache: EmbeddingCache 实例（用于缓存 query embedding）
        """
        self.embedder = embedder
        self._tokenizer = tokenizer
        self._has_embedding = embedder is not None
        self._chroma_persist_dir = chroma_persist_dir
        self._emb_cache = embedding_cache  # may be None

        # Chroma 惰性初始化
        self._chroma_client = None
        self._collection = None
        self._chroma_indexed = False  # 是否已将文档索引到 Chroma

    @property
    def has_embedding(self) -> bool:
        return self._has_embedding

    @property
    def has_chroma(self) -> bool:
        """Chroma 是否可用"""
        return self._chroma_client is not None

    def _ensure_chroma(self):
        """惰性初始化 Chroma client + collection"""
        if self._chroma_client is not None:
            return

        try:
            import chromadb

            if self._chroma_persist_dir:
                self._chroma_client = chromadb.PersistentClient(
                    path=self._chroma_persist_dir
                )
            else:
                self._chroma_client = chromadb.Client()

            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            # 检查是否有已持久化的数据
            if self._collection.count() > 0:
                self._chroma_indexed = True
                logger.info(
                    f"Chroma: 加载已有集合 '{self.COLLECTION_NAME}' "
                    f"({self._collection.count()} 篇文档)"
                )
            else:
                logger.info(f"Chroma: 创建新集合 '{self.COLLECTION_NAME}'")

        except ImportError:
            logger.warning("chromadb 未安装，向量检索降级为 Jaccard")
            self._chroma_client = None
        except Exception as e:
            logger.warning(f"Chroma 初始化失败，降级为 Jaccard: {e}")
            self._chroma_client = None

    def _doc_id(self, doc_or_text) -> str:
        """Generate stable Chroma ID from text + source (aligned with make_doc_id)."""
        if isinstance(doc_or_text, dict):
            text = doc_or_text.get("text", "")[:200]
            source = doc_or_text.get("meta", {}).get("source", "unknown")
            key = f"{source}|{text}"
        else:
            key = doc_or_text[:200]
        return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()[:16]

    # ===================== 索引操作 =====================

    def index(self, documents: List[Dict],
              embeddings: Optional[List[List[float]]] = None):
        """
        将文档 + embeddings 索引到 Chroma

        Args:
            documents: 文档列表 [{"text": "...", "meta": {...}}, ...]
            embeddings: 预计算的向量列表 (与 documents 等长)
        """
        self._ensure_chroma()
        if self._collection is None:
            logger.warning("Chroma 不可用，跳过向量索引")
            return

        if not documents:
            return

        # 清空旧数据
        if self._collection.count() > 0:
            # Chroma 不支持 where={}, 用 ids 删除全部
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)

        # 批量 upsert (dedup by ID within batch)
        texts = [d.get("text", "") for d in documents]
        ids = [self._doc_id(d) for d in documents]
        metas = [_flatten_meta(d.get("meta", {})) for d in documents]

        # Deduplicate: keep last occurrence if same ID appears multiple times
        seen = {}
        for i, cid in enumerate(ids):
            seen[cid] = i
        keep = sorted(seen.values())
        if len(keep) < len(ids):
            logger.info(f"Chroma: dedup {len(ids)} → {len(keep)} docs (removed {len(ids) - len(keep)} duplicate IDs)")
            ids = [ids[i] for i in keep]
            texts = [texts[i] for i in keep]
            metas = [metas[i] for i in keep]
            if embeddings:
                embeddings = [embeddings[i] for i in keep]

        kwargs = {"ids": ids, "documents": texts, "metadatas": metas}
        if embeddings:
            kwargs["embeddings"] = embeddings

        # Chroma batch size limit ~5000, split if needed
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            batch_kwargs = {k: v[i:end] for k, v in kwargs.items()}
            self._collection.add(**batch_kwargs)

        self._chroma_indexed = True
        logger.info(f"Chroma: 索引 {len(documents)} 篇文档到集合 '{self.COLLECTION_NAME}'")

    def add(self, documents: List[Dict],
            embeddings: Optional[List[List[float]]] = None,
            start_index: int = 0):
        """增量添加文档到 Chroma"""
        self._ensure_chroma()
        if self._collection is None or not documents:
            return

        ids = [self._doc_id(d) for d in documents]
        texts = [d.get("text", "") for d in documents]
        metas = [_flatten_meta(d.get("meta", {})) for d in documents]

        # Deduplicate within batch
        seen = {}
        for i, cid in enumerate(ids):
            seen[cid] = i
        keep = sorted(seen.values())
        if len(keep) < len(ids):
            logger.info(f"Chroma add: dedup {len(ids)} → {len(keep)} docs")
            ids = [ids[i] for i in keep]
            texts = [texts[i] for i in keep]
            metas = [metas[i] for i in keep]
            if embeddings:
                embeddings = [embeddings[i] for i in keep]

        kwargs = {"ids": ids, "documents": texts, "metadatas": metas}
        if embeddings:
            kwargs["embeddings"] = embeddings

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            batch_kwargs = {k: v[i:end] for k, v in kwargs.items()}
            self._collection.add(**batch_kwargs)

        logger.info(f"Chroma: 增量添加 {len(documents)} 篇文档")

    def remove(self, documents: List[Dict]):
        """从 Chroma 中删除指定文档"""
        self._ensure_chroma()
        if self._collection is None or not documents:
            return

        ids = [self._doc_id(d) for d in documents]
        try:
            self._collection.delete(ids=ids)
            logger.info(f"Chroma: 删除 {len(ids)} 篇文档")
        except Exception as e:
            logger.warning(f"Chroma delete failed (non-fatal): {e}")

    def clear(self):
        """清空 Chroma 集合"""
        self._ensure_chroma()
        if self._collection is None:
            return

        if self._collection.count() > 0:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)
        self._chroma_indexed = False
        logger.info("Chroma: 集合已清空")

    # ===================== 检索 =====================

    def search_embedding(self, documents: List[Dict], query: str, top_k: int,
                         doc_embeddings: Optional[List[List[float]]] = None,
                         cache_callback=None) -> List[Dict]:
        """
        语义检索 — 优先使用 Chroma ANN，降级为暴力余弦

        Args:
            documents: 文档列表
            query: 查询文本
            top_k: 返回数量
            doc_embeddings: 预计算的文档向量 (Chroma 不可用时降级用)
            cache_callback: 缓存回调 (降级路径用)
        """
        if not documents:
            return []

        # 计算 query embedding (走缓存，避免重复调 API)
        if self._emb_cache is not None:
            query_vec = self._emb_cache.embed_query(query, self.embedder)
        else:
            query_vec = self.embedder.embed_query(query)

        # 优先: Chroma ANN 检索
        self._ensure_chroma()
        if self._collection is not None and self._chroma_indexed:
            return self._query_chroma(documents, query_vec, top_k)

        # 降级: 暴力余弦 (Chroma 不可用或未索引)
        return self._search_brute_force(
            documents, query_vec, top_k, doc_embeddings, cache_callback
        )

    def _query_chroma(self, documents: List[Dict],
                      query_vec: List[float], top_k: int) -> List[Dict]:
        """通过 Chroma ANN 查询"""
        try:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, self._collection.count()),
                include=["distances", "documents"],
            )
        except Exception as e:
            logger.warning(f"Chroma query failed, falling back: {e}")
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0]

        # 构建文档 ID → 索引映射
        id_to_idx = {}
        for i, doc in enumerate(documents):
            id_to_idx[self._doc_id(doc.get("text", ""))] = i

        output = []
        for rank, (doc_id, dist) in enumerate(zip(ids, distances)):
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            # Chroma cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - distance/2
            score = max(0.0, 1.0 - dist / 2.0)
            output.append({
                **documents[idx],
                "score": score,
                "retriever": "vector",
                "rank": rank + 1,
            })

        return output

    def _search_brute_force(
        self, documents: List[Dict], query_vec: List[float],
        top_k: int,
        doc_embeddings: Optional[List[List[float]]] = None,
        cache_callback=None,
    ) -> List[Dict]:
        """暴力余弦检索 (Chroma 不可用时的降级路径)"""
        import math

        doc_vecs = doc_embeddings
        if not doc_vecs:
            texts = [d.get("text", "") for d in documents]
            doc_vecs = self.embedder.embed_documents(texts)
            if cache_callback:
                cache_callback(doc_vecs)

        scores = {}
        for i, dv in enumerate(doc_vecs):
            dot = sum(x * y for x, y in zip(query_vec, dv))
            norm_a = math.sqrt(sum(x * x for x in query_vec))
            norm_b = math.sqrt(sum(x * x for x in dv))
            scores[i] = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**documents[doc_id], "score": score, "retriever": "vector", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

    # ===================== Jaccard 回退 =====================

    def search_jaccard(self, documents: List[Dict], query: str, top_k: int,
                       tokenize_fn=None) -> List[Dict]:
        """无 API 时的 Jaccard 语义匹配"""
        if not documents:
            return []
        fn = tokenize_fn or self._tokenizer
        if fn is None:
            return []

        q_words = set(fn(query))
        if not q_words:
            return []

        scores = {}
        for doc_id, doc in enumerate(documents):
            dw = set(fn(doc.get("text", "")))
            if not dw:
                continue
            intersection = q_words & dw
            union = q_words | dw
            jaccard = len(intersection) / len(union) if union else 0
            length_penalty = min(1.0, 200 / max(len(dw), 1))
            scores[doc_id] = jaccard * 0.7 + length_penalty * 0.3

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**documents[doc_id], "score": score, "retriever": "vector_jaccard", "rank": i + 1}
            for i, (doc_id, score) in enumerate(ranked)
        ]
