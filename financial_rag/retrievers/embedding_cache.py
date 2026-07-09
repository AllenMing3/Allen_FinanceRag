"""
retrievers/embedding_cache.py — 内容 hash 级 embedding 缓存

目的：避免对相同文本重复调用 embedding API（省钱 + 省时间）。
机制：
- 对每段文本算 MD5 → 作为缓存 key
- 命中缓存时直接返回向量，不调 API
- 未命中的文本批量调 embed_documents，结果写回缓存
- 缓存持久化到磁盘 (JSON)，跨重启有效
- 支持 max_entries 限制，超出时 LRU 淘汰最旧条目
"""
import hashlib
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认缓存文件路径
_DEFAULT_CACHE_DIR = os.path.join("data", "knowledge_base")
_DEFAULT_CACHE_FILE = "embedding_cache.json"
_DEFAULT_MAX_ENTRIES = 5000


class EmbeddingCache:
    """
    内容 hash → embedding 向量的本地缓存。

    用法:
        cache = EmbeddingCache()           # 单例，自动加载磁盘缓存
        vecs = cache.embed_texts(texts, embedder)  # 替代 embedder.embed_documents()
        cache.save()                        # 持久化到磁盘
    """

    _instance: Optional["EmbeddingCache"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ):
        self._cache_dir = cache_dir
        self._cache_path = os.path.join(cache_dir, _DEFAULT_CACHE_FILE)
        self._max_entries = max_entries
        self._cache: Dict[str, List[float]] = {}
        self._access_order: Dict[str, float] = {}  # key → last access time (LRU)
        self._dirty = False  # 是否有未保存的变更
        self._load()

    @classmethod
    def get_instance(cls, cache_dir: str = _DEFAULT_CACHE_DIR) -> "EmbeddingCache":
        """获取全局单例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_dir=cache_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（测试用）"""
        cls._instance = None

    # ===================== 核心接口 =====================

    def embed_texts(self, texts: List[str], embedder) -> List[List[float]]:
        """
        带缓存的批量 embedding。

        替代直接调用 embedder.embed_documents(texts)。
        命中的走缓存，未命中的批量调 API 后写回缓存。
        """
        if not texts:
            return []

        # 1. 计算 hash + 查缓存
        keys = [self._hash(t) for t in texts]
        cached_vectors: Dict[str, List[float]] = {}
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, (key, text) in enumerate(zip(keys, texts)):
            if key in self._cache:
                cached_vectors[key] = self._cache[key]
                self._access_order[key] = time.time()
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        hits = len(texts) - len(uncached_texts)

        # 2. 未命中的调 API
        new_vectors: Dict[str, List[float]] = {}
        if uncached_texts and embedder is not None:
            batch_results = []
            for j in range(0, len(uncached_texts), 10):
                batch_results.extend(
                    embedder.embed_documents(uncached_texts[j:j + 10])
                )
            for idx, vec in zip(uncached_indices, batch_results):
                key = keys[idx]
                self._cache[key] = vec
                self._access_order[key] = time.time()
                new_vectors[key] = vec
            self._dirty = True
            self._evict_if_needed()

            logger.info(
                f"Embedding cache: {hits} hits, {len(uncached_texts)} misses "
                f"(cache size: {len(self._cache)})"
            )
        elif uncached_texts:
            logger.warning(
                f"Embedding cache: {hits} hits, {len(uncached_texts)} misses "
                f"but no embedder available"
            )

        # 3. 有变更时自动保存
        if self._dirty and new_vectors:
            self.save()

        # 4. 按原始顺序组装结果
        result = []
        for i, key in enumerate(keys):
            if key in self._cache:
                result.append(self._cache[key])
            else:
                # embedder 为 None 或调用失败，填充零向量
                dim = 1024  # 默认维度
                if result:
                    dim = len(result[0])
                result.append([0.0] * dim)

        return result

    def embed_query(self, query: str, embedder) -> List[float]:
        """带缓存的单条 query embedding（替代 embedder.embed_query）"""
        results = self.embed_texts([query], embedder)
        return results[0]

    # ===================== 持久化 =====================

    def save(self):
        """保存缓存到磁盘"""
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            data = {
                "version": 1,
                "entry_count": len(self._cache),
                "cache": self._cache,
            }
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            self._dirty = False
            logger.debug(
                f"Embedding cache saved: {len(self._cache)} entries "
                f"({os.path.getsize(self._cache_path) / 1024:.1f} KB)"
            )
        except Exception as e:
            logger.warning(f"Embedding cache save failed: {e}")

    def _load(self):
        """从磁盘加载缓存"""
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cache = data.get("cache", {})
            self._cache = cache
            # 初始化 access_order（所有条目设为同一时间）
            now = time.time()
            self._access_order = {k: now for k in self._cache}
            logger.info(
                f"Embedding cache loaded: {len(self._cache)} entries "
                f"from {self._cache_path}"
            )
        except Exception as e:
            logger.warning(f"Embedding cache load failed: {e}")
            self._cache = {}

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()
        self._dirty = False
        if os.path.exists(self._cache_path):
            try:
                os.remove(self._cache_path)
            except OSError:
                pass
        logger.info("Embedding cache cleared")

    # ===================== 内部工具 =====================

    @staticmethod
    def _hash(text: str) -> str:
        """内容 hash — 取前 500 字符计算 MD5（平衡速度和唯一性）"""
        return hashlib.md5(
            text[:500].encode("utf-8", errors="replace")
        ).hexdigest()

    def _evict_if_needed(self):
        """LRU 淘汰：超出 max_entries 时删除最久未访问的条目"""
        if len(self._cache) <= self._max_entries:
            return
        # 按访问时间排序，删除最旧的
        sorted_keys = sorted(self._access_order.items(), key=lambda x: x[1])
        to_remove = len(self._cache) - self._max_entries
        for key, _ in sorted_keys[:to_remove]:
            self._cache.pop(key, None)
            self._access_order.pop(key, None)
        logger.info(f"Embedding cache evicted {to_remove} entries (LRU)")

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def cache_path(self) -> str:
        return self._cache_path
