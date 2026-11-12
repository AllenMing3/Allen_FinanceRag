"""
Chroma ANN 路径质量锁 — vector_engine.py 接线正确性（此前零覆盖）。

背景（探针实锤）：
- P0：入库 ID 用 _doc_id(dict)（含 source），查询映射用 _doc_id(纯文本)，
  永不相等 → 有 embedder 时 ANN 恒返回 0 条，向量通道静默全灭。
- 分数公式 1-dist/2 给正交文档 0.5 幻影正分（实测 v3 向量归一化，应为 1-dist）。
- Chroma add 重复 ID 不报错留旧向量 → 应 upsert。
- _doc_id 不含 chunk_id → 同源同前缀 chunk 撞键被 dedup 丢数据。

设计：不 mock embedder —— 向量是显式数据 fixture，index(docs, embeddings=...)
入库 + 直调 _query_chroma(docs, query_vec, k) 检索，只测接线不测语义质量
（语义质量由真实 API 冒烟覆盖）。tmp 目录 ignore_cleanup_errors=True
（Chroma 文件锁会卡 Windows 清理）。
"""
import logging

import pytest

from financial_rag.retrievers.vector_engine import VectorEngine

pytestmark = pytest.mark.unit


@pytest.fixture
def chroma_dir(tmp_path):
    return str(tmp_path / "chroma")


def _mk_docs():
    """三篇文档 + 正交 one-hot 向量（数据 fixture，非 mock）。"""
    docs = [
        {"text": "商汤科技营收同比增长。", "meta": {"source": "s_a"}},
        {"text": "寒武纪芯片算力提升。", "meta": {"source": "s_b"}},
        {"text": "智谱完成新一轮融资。", "meta": {"source": "s_c"}},
    ]
    embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return docs, embs


class TestAnnRoundtrip:
    def test_ann_returns_mapped_hits(self, chroma_dir):
        """P0 复现锁：入库后 ANN 必须能映射回内存文档（修复前恒 0 条）。"""
        docs, embs = _mk_docs()
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, embs)
        res = ve._query_chroma(docs, [0.0, 1.0, 0.0], top_k=3)
        assert len(res) == 3, "ANN 命中必须映射回文档，不得静默全丢"
        assert res[0]["text"] == "寒武纪芯片算力提升。"
        assert res[0]["retriever"] == "vector" and res[0]["rank"] == 1

    def test_store_and_query_ids_aligned(self, chroma_dir):
        """入库 ID 与查询映射 ID 必须同源（防 dict/str 分支再度分裂）。"""
        docs, embs = _mk_docs()
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, embs)
        stored = set(ve._collection.get()["ids"])
        assert stored == {ve._doc_id(d) for d in docs}


class TestChunkCollision:
    def test_same_prefix_chunks_both_survive(self, chroma_dir):
        """同源同前 200 字、不同 chunk_id 的两个 chunk 必须双存活。"""
        prefix = "共同开头内容。" * 30  # >200 字，确保前缀截断后相同
        docs = [
            {"text": prefix + "结尾甲。",
             "meta": {"source": "same_src", "chunk_id": 0}},
            {"text": prefix + "结尾乙。",
             "meta": {"source": "same_src", "chunk_id": 1}},
        ]
        embs = [[1.0, 0.0], [0.0, 1.0]]
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, embs)
        assert ve._collection.count() == 2, "chunk 撞键会被 dedup 静默丢数据"
        res = ve._query_chroma(docs, [0.0, 1.0], top_k=2)
        assert {r["text"] for r in res} == {d["text"] for d in docs}


class TestScoreSemantics:
    def test_identical_orthogonal_opposite(self, chroma_dir):
        """归一化向量：相同→1.0、正交→0.0、相反→0.0（不得有幻影正分）。"""
        docs = [{"text": "目标文档。", "meta": {"source": "s"}}]
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, [[1.0, 0.0]])
        same = ve._query_chroma(docs, [1.0, 0.0], 1)
        orth = ve._query_chroma(docs, [0.0, 1.0], 1)
        opp = ve._query_chroma(docs, [-1.0, 0.0], 1)
        assert same[0]["score"] == pytest.approx(1.0, abs=1e-6)
        assert orth[0]["score"] == pytest.approx(0.0, abs=1e-6), \
            "正交文档不得有幻影正分（旧公式 1-dist/2 给 0.5）"
        assert opp[0]["score"] == 0.0


class TestUpsert:
    def test_readd_same_id_updates_vector(self, chroma_dir):
        """同 ID 重入库必须覆盖旧向量（Chroma add 重复 ID 静默留旧）。"""
        doc = {"text": "内容稳定不变。", "meta": {"source": "s"}}
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index([doc], [[1.0, 0.0]])
        ve.add([doc], [[0.0, 1.0]])
        assert ve._collection.count() == 1
        res = ve._query_chroma([doc], [0.0, 1.0], 1)
        assert res and res[0]["score"] == pytest.approx(1.0, abs=1e-6), \
            "重入库后旧向量残留 → 新向量检索不到"


class TestPersistenceReload:
    def test_new_engine_same_dir_searchable(self, chroma_dir):
        """持久化重载：同目录新引擎实例无需重索引即可 ANN 检索。"""
        docs, embs = _mk_docs()
        ve1 = VectorEngine(chroma_persist_dir=chroma_dir)
        ve1.index(docs, embs)
        ve2 = VectorEngine(chroma_persist_dir=chroma_dir)
        ve2._ensure_chroma()  # 生产路径由 search_embedding 先初始化，此处模拟
        res = ve2._query_chroma(docs, [0.0, 0.0, 1.0], top_k=1)
        assert res and res[0]["text"] == "智谱完成新一轮融资。"


class TestEnsureConsistent:
    def test_mismatch_triggers_rebuild(self, chroma_dir):
        """库与当前文档集失配（如 ID 方案迁移）→ 用传入向量全量重建。"""
        old_docs, old_embs = _mk_docs()
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(old_docs, old_embs)
        new_docs = [{"text": "全新文档。", "meta": {"source": "s_new"}}]
        ve.ensure_consistent(new_docs, [[1.0, 0.0, 0.0]])
        assert ve._collection.count() == 1
        res = ve._query_chroma(new_docs, [1.0, 0.0, 0.0], 1)
        assert res and res[0]["text"] == "全新文档。"

    def test_consistent_is_noop(self, chroma_dir):
        """集合一致时不得触发重建（返回 False，向量原样）。"""
        docs, embs = _mk_docs()
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, embs)
        assert ve.ensure_consistent(docs, embs) is False
        assert ve._collection.count() == 3


class TestUnmappedObservability:
    def test_partial_miss_warns(self, chroma_dir, caplog):
        """部分命中未映射（库含陈旧条目）→ warning 含 miss 计数，不得静默。"""
        docs, embs = _mk_docs()
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, embs)
        with caplog.at_level(logging.WARNING,
                             logger="financial_rag.retrievers.vector_engine"):
            res = ve._query_chroma(docs[:1], [1.0, 0.0, 0.0], top_k=3)
        assert len(res) == 1  # 可映射的照常返回
        assert any("未映射" in r.message for r in caplog.records)

    def test_all_miss_logs_error(self, chroma_dir, caplog):
        """全部未映射 = 库失配信号 → error 级日志 + 返回空。"""
        docs, embs = _mk_docs()
        ve = VectorEngine(chroma_persist_dir=chroma_dir)
        ve.index(docs, embs)
        other = [{"text": "库外文档。", "meta": {"source": "outside"}}]
        with caplog.at_level(logging.ERROR,
                             logger="financial_rag.retrievers.vector_engine"):
            res = ve._query_chroma(other, [1.0, 0.0, 0.0], top_k=3)
        assert res == []
        assert any(r.levelno == logging.ERROR for r in caplog.records)