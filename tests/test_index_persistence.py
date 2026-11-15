"""
索引持久化回归 — 派生索引与代码默认值的一致性（阶段0 两处发现）。

背景（真实数据解剖发现）：
  Finding 2（代码缺陷）: load_index 曾用磁盘 config 里的旧通道权重
    （如已修复的 0.3/0.7 倒挂）覆盖代码默认 0.5/0.5，悄悄违反
    test_fusion_step4 守的 bm25_weight >= vector_weight 不变量。
    根因：通道权重是代码级调参，却被持久化进数据索引再盲目读回 → 漂移。
  Finding 1（机制锁定）: 磁盘 BM25 行数可能与真相源不一致（测试残留），
    但 load_index 会 _bm25.build(loaded_docs) 从加载文档重建 → 自愈。
    本文件锁定该自愈行为，防回归。

离线约束：conftest 已删 DASHSCOPE_API_KEY；HybridRetriever(embedder=None,
reranker=None) 纯本地 BM25+Jaccard，不触 Chroma 向量；bm25/index 全走 tmp_path
隔离，不碰真实 data/knowledge_base。用旧权重正常 save 复现漂移，不 mock 逻辑。
"""
import pytest

from financial_rag.retrievers.hybrid_engine import HybridRetriever

pytestmark = [pytest.mark.unit, pytest.mark.critical]


_DOCS = [
    {"text": "贵州茅台2024年年度报告，营业收入1741亿元。", "meta": {"source": "s1"}},
    {"text": "商汤科技生成式AI业务收入占比达60%。", "meta": {"source": "s2"}},
    {"text": "宁德时代动力电池装机量全球第一。", "meta": {"source": "s3"}},
]


def _make_retriever(tmp_path):
    """纯本地检索器，bm25/chroma 全隔离到 tmp_path。"""
    return HybridRetriever(
        embedder=None, reranker=None,
        bm25_db_path=str(tmp_path / "bm25_index.db"),
        chroma_persist_dir=str(tmp_path / "chroma"),
    )


class TestConfigDriftRegression:
    """Finding 2：通道权重归代码所有，磁盘旧权重不得覆盖代码默认。"""

    def test_load_index_keeps_code_weights_not_stale_disk(self, tmp_path):
        # 1. 模拟旧代码：用过期倒挂权重(0.3/0.7)正常建索引并落盘。
        #    save_index 按当时权重算 checksum → 文件完整(checksum有效)但语义过期。
        #    注意：不能事后篡改 config（会触发 checksum 完整性保护，偏离真实漂移路径）。
        r1 = _make_retriever(tmp_path)
        r1.bm25_weight = 0.3
        r1.vector_weight = 0.7
        r1.index(_DOCS, precompute_embeddings=False)
        idx_path = str(tmp_path / "kb_index.json")
        r1.save_index(idx_path)

        # 2. 全新检索器（代码默认）加载该旧索引：checksum 有效，但权重过期
        r2 = _make_retriever(tmp_path)
        default_bm25 = r2.bm25_weight
        default_vector = r2.vector_weight
        r2.load_index(idx_path)

        # 4. 权重必须保持代码默认，不被磁盘旧值覆盖
        assert r2.bm25_weight == default_bm25, (
            f"load_index 用磁盘旧权重覆盖了代码默认: "
            f"{r2.bm25_weight} != {default_bm25}")
        assert r2.vector_weight == default_vector, (
            f"load_index 用磁盘旧权重覆盖了代码默认: "
            f"{r2.vector_weight} != {default_vector}")
        # 5. 强通道权重不得倒挂（test_fusion_step4 同款不变量）
        assert r2.bm25_weight >= r2.vector_weight, (
            f"加载旧索引后 RRF 权重倒挂: "
            f"bm25={r2.bm25_weight} < vector={r2.vector_weight}")


class TestBM25SelfHealOnLoad:
    """Finding 1：load_index 从加载文档重建 BM25，磁盘陈旧自愈。"""

    def test_load_index_rebuilds_bm25_from_loaded_docs(self, tmp_path):
        # 1. 建索引（3 篇）并落盘
        r1 = _make_retriever(tmp_path)
        r1.index(_DOCS, precompute_embeddings=False)
        idx_path = str(tmp_path / "kb_index.json")
        r1.save_index(idx_path)
        assert r1._bm25.doc_count == 3

        # 2. 模拟磁盘 BM25 陈旧：清空（只剩 0 行，真相源/索引文件仍是 3 篇）
        r1._bm25.clear()
        assert r1._bm25.doc_count == 0

        # 3. 全新检索器加载索引 → 应从加载的 3 篇重建 BM25
        r2 = _make_retriever(tmp_path)
        r2.load_index(idx_path)

        # 4. BM25 自愈到与文档集一致
        assert len(r2.documents) == 3
        assert r2._bm25.doc_count == 3, (
            f"load_index 未从真相源重建 BM25: "
            f"doc_count={r2._bm25.doc_count} != 3")