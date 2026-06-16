"""
数据编排器 — Data Orchestrator

管理多个 KnowledgePool，每个 Pool 是一个独立的 HybridRetriever。
自动根据 DocTypeClassifier 将文档路由到对应的 Pool。
支持跨 Pool 检索和加权融合。

架构:
    原始文档
        │
        ▼
    ┌──────────────────────────────────────┐
    │  TextPreprocessor (清洗)              │
    │  RelevanceGate (相关性门控)           │
    │  DocTypeClassifier (分类)            │
    └──────────────┬───────────────────────┘
                   │
        ┌──────────┼──────────┬──────────────┐
        ▼          ▼          ▼              ▼
   financial_   news       macro_        general
   report                 data

用法:
    orchestrator = DataOrchestrator()
    orchestrator.ingest(documents)
    results = orchestrator.search("茅台2024年营收", top_k=10)
    results = orchestrator.cross_search("降准影响", primary="macro_data", secondary=["news"])
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 默认 Pool 名称
DEFAULT_POOLS = ["financial_report", "news", "macro_data", "general"]

# doc_type → pool 名称映射
DOC_TYPE_TO_POOL = {
    "financial_report": "financial_report",
    "news": "news",
    "macro_data": "macro_data",
    "query": "general",
    "other": "general",
}


@dataclass
class IngestStats:
    """摄取统计"""
    total_docs: int = 0
    cleaned: int = 0
    rejected: int = 0
    routed: Dict[str, int] = field(default_factory=dict)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def summary(self) -> str:
        route_str = ", ".join(f"{k}={v}" for k, v in self.routed.items() if v > 0)
        reject_str = ", ".join(f"{k}={v}" for k, v in self.rejection_reasons.items() if v > 0)
        return (
            f"IngestStats: total={self.total_docs}, cleaned={self.cleaned}, "
            f"rejected={self.rejected} [{reject_str}], routed=[{route_str}], "
            f"elapsed={self.elapsed_ms:.0f}ms"
        )


@dataclass
class KnowledgePool:
    """知识池 — 一个命名的 HybridRetriever 包装"""
    name: str
    retriever: Any = None  # HybridRetriever
    doc_count: int = 0
    last_updated: str = ""

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict]:
        """在池中检索"""
        if self.retriever is None or self.doc_count == 0:
            return []
        return self.retriever.search(query, top_k=top_k, **kwargs)


class DataRouter:
    """文档路由器 — 根据 doc_type 决定放入哪个 Pool"""

    def __init__(self, pool_names: Optional[List[str]] = None):
        self._pool_names = set(pool_names or DEFAULT_POOLS)
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            from financial_rag.retrievers.preprocessor import DocTypeClassifier
            self._classifier = DocTypeClassifier()
        return self._classifier

    def route(self, doc: Dict) -> str:
        """
        路由单个文档到目标 Pool。

        优先使用 doc.meta.doc_type (已由 IngestionAgent 设置)。
        若无 doc_type，使用 DocTypeClassifier 分类。
        """
        meta = doc.get("meta", doc.get("metadata", {}))
        doc_type = meta.get("doc_type", "")

        if not doc_type:
            text = doc.get("text", "")
            if text:
                result = self._get_classifier().classify(text)
                doc_type = result.get("doc_type", "other")
                # 回写 meta 供后续使用
                meta["doc_type"] = doc_type
                if "meta" not in doc:
                    doc["meta"] = meta

        pool_name = DOC_TYPE_TO_POOL.get(doc_type, "general")

        # 如果目标 Pool 不存在，降级到 general
        if pool_name not in self._pool_names:
            pool_name = "general"

        return pool_name


class DataOrchestrator:
    """
    数据编排器 — 管理多个 KnowledgePool

    职责:
    1. 维护多个命名 Pool (每个 Pool 有自己的 HybridRetriever)
    2. 摄取文档: 清洗 → 相关性门控 → 分类 → 路由到对应 Pool
    3. 跨 Pool 检索: 搜索多个 Pool，加权融合结果
    """

    def __init__(
        self,
        pool_names: Optional[List[str]] = None,
        retriever_factory: Any = None,
    ):
        """
        Args:
            pool_names: Pool 名称列表，默认 ["financial_report", "news", "macro_data", "general"]
            retriever_factory: 可调用对象，返回 HybridRetriever 实例。
                              默认创建无 embedder/reranker 的基础 Retriever。
        """
        self._pool_names = pool_names or list(DEFAULT_POOLS)
        self._retriever_factory = retriever_factory or self._default_retriever
        self._router = DataRouter(self._pool_names)

        # 创建 Pools
        self.pools: Dict[str, KnowledgePool] = {}
        for name in self._pool_names:
            self.pools[name] = KnowledgePool(
                name=name,
                retriever=self._retriever_factory(),
            )

        # 预处理器 (延迟导入)
        self._preprocessor = None
        self._gate = None

    @staticmethod
    def _default_retriever():
        """创建默认的基础 HybridRetriever (无 API 依赖)"""
        from financial_rag.retrievers.retriever import HybridRetriever
        return HybridRetriever()

    def _get_preprocessor(self):
        if self._preprocessor is None:
            from financial_rag.retrievers.preprocessor import TextPreprocessor
            self._preprocessor = TextPreprocessor()
        return self._preprocessor

    def _get_gate(self):
        if self._gate is None:
            from financial_rag.retrievers.preprocessor import RelevanceGate
            self._gate = RelevanceGate()
        return self._gate

    # ===================== Pool 管理 =====================

    def add_pool(self, name: str, retriever: Any = None) -> KnowledgePool:
        """添加自定义 Pool"""
        pool = KnowledgePool(
            name=name,
            retriever=retriever or self._retriever_factory(),
        )
        self.pools[name] = pool
        self._router._pool_names.add(name)
        logger.info(f"新增 Pool: {name}")
        return pool

    def remove_pool(self, name: str):
        """移除 Pool"""
        if name in self.pools:
            del self.pools[name]
            self._router._pool_names.discard(name)

    # ===================== 摄取 =====================

    def ingest(self, documents: List[Dict]) -> IngestStats:
        """
        摄取文档到对应的 Pool。

        流程: 清洗 → 相关性门控 → 分类路由 → 按 Pool 分组 → 批量索引

        Returns:
            IngestStats: 摄取统计信息
        """
        t0 = time.time()
        stats = IngestStats(total_docs=len(documents))
        preprocessor = self._get_preprocessor()
        gate = self._get_gate()

        # 按 Pool 分组
        pool_docs: Dict[str, List[Dict]] = {name: [] for name in self.pools}

        for doc in documents:
            text = doc.get("text", "")
            if not text or not text.strip():
                stats.rejected += 1
                stats.rejection_reasons["empty_text"] = stats.rejection_reasons.get("empty_text", 0) + 1
                continue

            # 1. 清洗
            clean_text = preprocessor.process(text)
            if not clean_text.strip():
                stats.rejected += 1
                stats.rejection_reasons["cleaning_emptied"] = stats.rejection_reasons.get("cleaning_emptied", 0) + 1
                logger.debug(f"DataOrchestrator: doc rejected (cleaning emptied text, source={doc.get('meta', {}).get('source', '?')})")
                continue
            doc["text"] = clean_text
            stats.cleaned += 1

            # 2. 相关性门控
            passed, reason, kw_count = gate.check(clean_text)
            if not passed:
                meta = doc.get("meta", {})
                meta["_rejected"] = True
                meta["_reject_reason"] = reason
                doc["meta"] = meta
                stats.rejected += 1
                stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1
                logger.debug(f"DataOrchestrator: doc rejected ({reason}, source={meta.get('source', '?')})")
                continue

            # 3. 路由
            pool_name = self._router.route(doc)
            pool_docs[pool_name].append(doc)

        # 4. 批量索引到各 Pool
        for pool_name, docs in pool_docs.items():
            if not docs:
                continue
            pool = self.pools[pool_name]
            if pool.retriever is None:
                pool.retriever = self._retriever_factory()

            if pool.doc_count == 0:
                pool.retriever.index(docs, precompute_embeddings=False, use_chunker=False)
            else:
                pool.retriever.add(docs, use_chunker=False)

            pool.doc_count += len(docs)
            pool.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
            stats.routed[pool_name] = len(docs)
            logger.info(f"Pool [{pool_name}]: +{len(docs)} docs (total: {pool.doc_count})")

        stats.elapsed_ms = (time.time() - t0) * 1000
        logger.info(stats.summary())
        return stats

    # ===================== 检索 =====================

    def search(
        self,
        query: str,
        pool_names: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        跨 Pool 检索。

        Args:
            query: 查询文本
            pool_names: 要搜索的 Pool 列表，默认搜索所有非空 Pool
            weights: Pool 权重 {"financial_report": 2.0, "news": 1.0, ...}
            top_k: 返回条数

        Returns:
            融合后的排序结果
        """
        targets = pool_names or [n for n, p in self.pools.items() if p.doc_count > 0]
        default_weight = 1.0

        all_results = []
        for pool_name in targets:
            pool = self.pools.get(pool_name)
            if not pool or pool.doc_count == 0:
                continue

            w = (weights or {}).get(pool_name, default_weight)
            results = pool.search(query, top_k=top_k * 2)

            # 标记来源 Pool 并应用权重
            for r in results:
                r["_pool"] = pool_name
                r["_pool_weight"] = w
                r["score"] = r.get("score", 0) * w
                # 重建 rank (按加权分数)
            all_results.extend(results)

        # 按加权分数排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 去重 (按 text hash)
        seen = set()
        deduped = []
        for r in all_results:
            text_hash = hash(r.get("text", ""))
            if text_hash not in seen:
                seen.add(text_hash)
                deduped.append(r)

        # 重建 rank
        for i, r in enumerate(deduped[:top_k]):
            r["rank"] = i + 1

        return deduped[:top_k]

    def cross_search(
        self,
        query: str,
        primary: str,
        secondary: Optional[List[str]] = None,
        top_k: int = 10,
        secondary_top_k: int = 3,
    ) -> List[Dict]:
        """
        跨域检索: 主 Pool 结果 + 辅 Pool 补充。

        Args:
            query: 查询文本
            primary: 主 Pool 名称
            secondary: 辅助 Pool 列表，默认除主 Pool 外所有非空 Pool
            top_k: 主 Pool 返回条数
            secondary_top_k: 每个辅 Pool 补充条数

        Returns:
            主 Pool 结果在前，辅 Pool 结果追加 (带 _pool 标记)
        """
        # 主 Pool
        primary_pool = self.pools.get(primary)
        if not primary_pool or primary_pool.doc_count == 0:
            # 降级到普通搜索
            return self.search(query, top_k=top_k)

        results = primary_pool.search(query, top_k=top_k)
        for r in results:
            r["_pool"] = primary
            r["_cross_role"] = "primary"

        # 辅助 Pool
        if secondary is None:
            secondary = [n for n, p in self.pools.items()
                        if n != primary and p.doc_count > 0]

        seen_texts = {hash(r.get("text", "")) for r in results}

        for pool_name in secondary:
            pool = self.pools.get(pool_name)
            if not pool or pool.doc_count == 0:
                continue

            sec_results = pool.search(query, top_k=secondary_top_k * 2)
            added = 0
            for r in sec_results:
                text_hash = hash(r.get("text", ""))
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    r["_pool"] = pool_name
                    r["_cross_role"] = "secondary"
                    results.append(r)
                    added += 1
                    if added >= secondary_top_k:
                        break

        return results

    # ===================== 统计 =====================

    def get_stats(self) -> Dict:
        """获取所有 Pool 的统计信息"""
        stats = {}
        total = 0
        for name, pool in self.pools.items():
            stats[name] = {
                "doc_count": pool.doc_count,
                "last_updated": pool.last_updated,
                "has_retriever": pool.retriever is not None,
            }
            total += pool.doc_count
        stats["_total"] = total
        stats["_pool_count"] = len(self.pools)
        return stats

    def clear(self, pool_names: Optional[List[str]] = None):
        """清空指定 Pool 或所有 Pool"""
        targets = pool_names or list(self.pools.keys())
        for name in targets:
            pool = self.pools.get(name)
            if pool and pool.retriever:
                pool.retriever.clear()
                pool.doc_count = 0
                pool.last_updated = ""
                logger.info(f"Pool [{name}] 已清空")
