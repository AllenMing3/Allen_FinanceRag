"""
Test Step 1 入库 — 文档 → chunk 切分 → 索引（RAG 流水线第一步）。

这是"沿 RAG 每一步建探针"的第一个永久资产：把一次性探针的结论钉成不可回退的用例。
按项目四条测试标准组织：
  ① 证明效果  ② 可观测  ③ 一致性  ④ 回归锁定（conftest 离线可测性修复）

评估基准数据：financial_rag/eval/golden/eval_kb.jsonl（18 篇 AI 领域短文，均 <500 字）

对应生产代码：financial_rag/retrievers/factory.py + hybrid_engine.py + chunker.py
"""
from pathlib import Path

import pytest

from financial_rag.retrievers.factory import create_hybrid_retriever

_GOLDEN_KB = Path(__file__).resolve().parents[1] / "financial_rag" / "eval" / "golden" / "eval_kb.jsonl"
_SKIP_THRESHOLD = 500  # 与 factory 装配的 TextChunker(skip_threshold=500) 对齐


def _load_golden_docs():
    """把 golden eval_kb.jsonl 转成入库入参形状 {"text", "meta":{doc_id,...}}。"""
    import json
    docs = []
    with open(_GOLDEN_KB, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append({
                "text": d["text"],
                "meta": {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "doc_type": d["doc_type"],
                    "source": d["source"],
                },
            })
    return docs


@pytest.fixture
def golden_docs():
    return _load_golden_docs()


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    """把 config 数据目录指向临时路径，隔离 chroma/bm25 落盘，不碰生产库。

    自包含——不依赖 conftest 的 fixture（避免被上游回退时本资产跟着失效）。
    factory 在调用时读 config.kb_dir，故 monkeypatch 在 create_hybrid_retriever 前生效即可。
    """
    from financial_rag.config import config
    kb = tmp_path / "knowledge_base"
    data = tmp_path / "financial"
    out = tmp_path / "output"
    for d in (kb, data, out):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "kb_dir", str(kb))
    monkeypatch.setattr(config, "data_dir", str(data))
    monkeypatch.setattr(config, "output_dir", str(out))
    return kb


@pytest.fixture
def indexed_retriever(isolated_kb, golden_docs):
    """在临时数据目录里建检索器并灌入 golden 文档。

    conftest 的 autouse fixture 已删 DASHSCOPE_API_KEY → 纯本地 BM25+Jaccard，
    无 embedding/网络，确定性可复现。isolated_kb 把 config.kb_dir 指向临时路径，
    chroma/bm25 落盘不碰生产库。
    """
    r = create_hybrid_retriever()
    r.index(golden_docs)
    return r


# ===========================================================================
# ① 证明效果：入库产出符合"短文不切"的设计契约
# ===========================================================================
@pytest.mark.integration
class TestIngestEffect:
    def test_all_short_docs_indexed_one_chunk_each(self, indexed_retriever, golden_docs):
        # golden 全部 <500 字 → 每篇整篇入库，18 篇应产出恰好 18 个 chunk
        assert all(len(d["text"]) <= _SKIP_THRESHOLD for d in golden_docs), "前提：golden 应全为短文"
        assert len(indexed_retriever.documents) == len(golden_docs)

    def test_every_doc_id_survives_ingest(self, indexed_retriever, golden_docs):
        # doc_id 是后续检索评估对齐 relevant_doc_ids 的命脉，入库不得丢
        expected = {d["meta"]["doc_id"] for d in golden_docs}
        actual = {c["meta"].get("doc_id") for c in indexed_retriever.documents}
        assert actual == expected, f"doc_id 集合不一致，缺失 {expected - actual}"


# ===========================================================================
# ② 可观测：每个 chunk 暴露可诊断的 meta 字段
# ===========================================================================
@pytest.mark.integration
class TestIngestObservability:
    def test_chunk_meta_has_observable_fields(self, indexed_retriever):
        for c in indexed_retriever.documents:
            m = c["meta"]
            assert "chunk_id" in m and "chunk_count" in m, f"缺可观测字段: {m.keys()}"
            assert m["chunk_count"] == 1  # 短文路径
            assert m.get("source_id"), "source_id 应回退到 title，不得为空"


# ===========================================================================
# ③ 一致性：文本不丢、重复入库结果确定
# ===========================================================================
@pytest.mark.integration
class TestIngestConsistency:
    def test_chunk_text_equals_source(self, indexed_retriever, golden_docs):
        # 短文不切 → chunk 文本应 == 原文(strip 后)，正文零改动
        orig = {d["meta"]["doc_id"]: d["text"].strip() for d in golden_docs}
        for c in indexed_retriever.documents:
            assert c["text"].strip() == orig[c["meta"]["doc_id"]]

    def test_reindex_is_deterministic(self, isolated_kb, golden_docs):
        # 同输入二次入库，文档集合完全一致（无随机、无时间依赖）
        r1 = create_hybrid_retriever()
        r1.index(golden_docs)
        r2 = create_hybrid_retriever()
        r2.index(golden_docs)
        assert r1.documents == r2.documents


# ===========================================================================
# ④ 回归锁定：force_local 离线可测性修复
# ===========================================================================
@pytest.mark.regression
class TestOfflineLocalLock:
    def test_retriever_is_local_despite_dotenv_key(self, isolated_kb):
        """钉死发现1的修复：conftest 的 _force_local_retrieval(autouse) 在测试期删除了
        DASHSCOPE_API_KEY（它会被 config 导入期 load_dotenv(override=True) 注入）。
        因此即便生产 .env 配了真 key，测试期建的检索器也必须是本地模式。
        退回旧状态（conftest 无该 fixture）→ env 残留真 key → embedder 非空即红。
        """
        import os
        assert "DASHSCOPE_API_KEY" not in os.environ, \
            "conftest 应在测试期清除 DASHSCOPE_API_KEY（离线可测性修复）"
        r = create_hybrid_retriever()
        assert r.embedder is None, "测试期不得装配真实 embedder"
        assert r.reranker is None, "测试期不得装配真实 reranker"
