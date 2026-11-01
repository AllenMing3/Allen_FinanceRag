"""
Test Step 1 入库 — 文档 → chunk 切分 → 索引（RAG 流水线第一步）。

这是“沿 RAG 每一步建探针”的第一个永久资产：把一次性探针的结论钉成不可回退的用例。
按项目四条测试标准组织：
  ① 证明效果  ② 可观测  ③ 一致性  ④ 回归锁定

Step 3 教训强化：
  - 不能只检查 Python 列表，必须验证 BM25 FTS5 实际索引了数据
  - 不能只测短文路径，必须覆盖长文切分（chunker 激活）
  - 不能只测 doc_id 存在，必须验证“可搜索到”

评估基准数据：financial_rag/eval/golden/eval_kb.jsonl（22 篇，含 4 篇非 AI 对照，均 <500 字）

对应生产代码：financial_rag/retrievers/factory.py + hybrid_engine.py + chunker.py + bm25_engine.py
"""
from pathlib import Path

import pytest

from financial_rag.retrievers.factory import create_hybrid_retriever
from financial_rag.retrievers.bm25_engine import BM25Engine
from financial_rag.retrievers.chunker import TextChunker

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


# ===========================================================================
# Step 3 教训强化：数据不丢 + 跨通道一致 + 长文切分完整性 + 增量可靠
# ===========================================================================

# 长文档（>500 字）用于触发 chunker 切分
# doc_type="news" → chunk_size=500；578 字 > 500 → 至少切出 2 chunks
_LONG_DOC = {
    "text": (
        "商汤科技发布2024年年度报告。"
        "全年实现营业收入50.3亿元，同比增长36.4%。"
        "其中生成式AI业务收入占比达60%，成为核心增长引擎。"
        "公司全年净亏损收窄至12.8亿元，同比减亏42%。"
        "研发投入达45亿元，占营收比例89.5%。"
        "日日新大模型API日均调用量突破2000万次，同比增长400%。"
        "SenseCore大装置算力规模达4万卡A100 GPU，算力利用率提升至85%。"
        "推理成本降至0.5元每百万token，较上年下降65%。"
        "公司与超过3000家企业客户建立合作，覆盖金融、医疗、制造、零售等行业。"
        "商汤科技董事长表示，公司将继续坚持原创技术研究，"
        "在生成式AI和多模态领域持续投入，"
        "力争在未来三年内实现盈亏平衡。"
        "公司计划进一步扩大算力基础设施投资，"
        "并在东南亚和中东市场布局海外业务。"
        "此外，商汤科技在自动驾驶、智能医疗、智慧城市等垂直领域也取得了显著进展。"
        "公司旗下商汤绝影自动驾驶平台已与超过30家车企合作，"
        "覆盖L2+到L4级别自动驾驶解决方案。"
        "在智能医疗领域，商汤科技与多家三甲医院合作，"
        "推动AI辅助诊断系统落地，覆盖影像诊断、病理分析等场景。"
        "在智慧城市领域，商汤科技已服务超过100个城市的数字化转型，"
        "提供视频分析、交通管理、应急指挥等解决方案。"
        "公司表示将继续加大在AGI通用人工智能领域的研究投入，"
        "目标是构建具备多模态理解与推理能力的下一代AI系统。"
    ),
    "meta": {"doc_id": "d_sensetime_long", "source": "eval_kb", "doc_type": "news"},
}


@pytest.mark.integration
class TestIngestDataCompleteness:
    """① 证明效果：入库后数据在 BM25 中实际可检索（不只是 Python 列表）"""

    def test_bm25_doc_count_matches_input(self, indexed_retriever, golden_docs):
        """BM25 FTS5 实际索引数 == 输入文档数。

        Step3 教训：Python 列表 documents 不代表 BM25 真的索引了。
        _insert_batch 去重可能静默丢弃数据。
        """
        assert indexed_retriever._bm25.doc_count == len(golden_docs)

    def test_every_doc_searchable_in_bm25(self, indexed_retriever, golden_docs):
        """每篇文档都能通过 BM25 搜索到（用其标题作为 query）。

        Step3 教训：doc_id 存在于 Python 列表 ≠ 可被检索到。
        """
        for doc in golden_docs:
            title = doc["meta"]["title"]
            results = indexed_retriever._bm25.search(
                indexed_retriever.documents, title, top_k=len(golden_docs)
            )
            found_ids = {r["meta"]["doc_id"] for r in results}
            assert doc["meta"]["doc_id"] in found_ids, (
                f"{doc['meta']['doc_id']} 在 BM25 中搜索不到 (query='{title}')"
            )

    def test_cross_channel_parity(self, indexed_retriever, golden_docs):
        """BM25 和 Jaccard 对同一 query 都能返回结果（两通道都有数据）。"""
        query = "商汤科技营收"
        bm25_res = indexed_retriever._bm25.search(
            indexed_retriever.documents, query, top_k=3
        )
        jaccard_res = indexed_retriever._vector.search_jaccard(
            indexed_retriever.documents, query, top_k=3,
            tokenize_fn=indexed_retriever._bm25.tokenize,
        )
        assert len(bm25_res) > 0, "BM25 通道无结果"
        assert len(jaccard_res) > 0, "Jaccard 通道无结果"


@pytest.mark.regression
class TestLongDocChunkIntegrity:
    """④ 回归锁定：长文档切分后 ALL chunks 在 BM25 中存活。

    Step3 发现的 bug：chunker 切分后所有 chunk 继承同一 doc_id，
    BM25 doc_key 碰撞 → _insert_batch 去重只保留最后一个 chunk。
    修复后：doc_key 生成复合 key {doc_id}_c{chunk_id}。
    """

    def test_long_doc_produces_multiple_chunks(self, isolated_kb):
        """前提验证：>500 字文档确实被切分为多 chunk。"""
        assert len(_LONG_DOC["text"]) > _SKIP_THRESHOLD
        r = create_hybrid_retriever()
        r.index([_LONG_DOC])
        # chunker 激活后应产出 > 1 个 chunk
        assert len(r.documents) > 1, (
            f"长文档({len(_LONG_DOC['text'])}字)应被切分，"
            f"实际只有 {len(r.documents)} chunk"
        )

    def test_all_chunks_indexed_in_bm25(self, isolated_kb):
        """核心回归锁：所有 chunk 在 BM25 FTS5 中存活（不是只检查 Python 列表）。"""
        r = create_hybrid_retriever()
        r.index([_LONG_DOC])
        total_chunks = len(r.documents)
        assert total_chunks > 1
        # BM25 实际索引数必须 == chunk 数（修复前会 < chunk 数）
        assert r._bm25.doc_count == total_chunks, (
            f"BM25 只索引了 {r._bm25.doc_count}/{total_chunks} chunks"
            f"（doc_key 碰撞导致静默丢失）"
        )

    def test_each_chunk_searchable(self, isolated_kb):
        """每个 chunk 都能通过 BM25 搜索到（用其自身文本片段作为 query）。"""
        r = create_hybrid_retriever()
        r.index([_LONG_DOC])
        for chunk in r.documents:
            # 用 chunk 文本的前 20 字作为 query，应该能找到自己
            probe = chunk["text"][:20]
            results = r._bm25.search(r.documents, probe, top_k=len(r.documents))
            found_texts = {res["text"][:50] for res in results}
            assert chunk["text"][:50] in found_texts, (
                f"chunk_id={chunk['meta'].get('chunk_id')} 在 BM25 中搜索不到"
            )


@pytest.mark.integration
class TestIncrementalIngest:
    """③ 一致性：增量 add 后新文档立即可搜索。"""

    def test_add_makes_doc_searchable(self, indexed_retriever):
        """add 一篇新文档后，BM25 能立即搜索到它。"""
        new_doc = {
            "text": "宁德时代发布2024年年报，全年营收4009亿元，净利润507亿元。",
            "meta": {"doc_id": "d_catl_2024", "source": "eval_kb", "doc_type": "financial_report"},
        }
        before = indexed_retriever._bm25.doc_count
        indexed_retriever.add([new_doc])
        after = indexed_retriever._bm25.doc_count
        assert after == before + 1, f"add 后 BM25 doc_count 未增加: {before} -> {after}"
        # 可搜索验证
        results = indexed_retriever._bm25.search(
            indexed_retriever.documents, "宁德时代营收", top_k=5
        )
        found_ids = {r["meta"]["doc_id"] for r in results}
        assert "d_catl_2024" in found_ids

    def test_remove_makes_doc_unsearchable(self, indexed_retriever, golden_docs):
        """remove 后文档从 BM25 消失。

        remove() 接收位置索引 List[int]，不是文档对象。
        """
        target = golden_docs[0]
        target_id = target["meta"]["doc_id"]
        # 找到 target 在 indexed_retriever.documents 中的位置索引
        target_idx = next(
            i for i, d in enumerate(indexed_retriever.documents)
            if d["meta"].get("doc_id") == target_id
        )
        indexed_retriever.remove([target_idx])
        results = indexed_retriever._bm25.search(
            indexed_retriever.documents, target["meta"]["title"], top_k=len(golden_docs)
        )
        found_ids = {r["meta"]["doc_id"] for r in results}
        assert target_id not in found_ids, f"{target_id} 应已被移除但仍可搜索到"

    def test_remove_by_doc_ids_removes_all_chunks(self, isolated_kb):
        """核心回归锁：多 chunk 文档按 doc_id 删除时，ALL chunks 从 BM25 消失。

        暴露的生产 bug：kb_router 用原始文档位置索引调 remove()，
        但 retriever.documents 是切分后的列表，位置对不上 → 删错 chunk。
        修复：新增 remove_by_doc_ids() 按 meta.doc_id 匹配，不依赖位置。
        """
        r = create_hybrid_retriever()
        # 灌入长文档（news, 578字 > chunk_size=500 → 多 chunk）+ 一篇短文档
        short_doc = {
            "text": "宁德时代发布2024年年报。",
            "meta": {"doc_id": "d_catl", "source": "test", "doc_type": "news"},
        }
        r.index([_LONG_DOC, short_doc])
        # 确认长文档被切分
        long_chunks = [d for d in r.documents if d["meta"].get("doc_id") == "d_sensetime_long"]
        assert len(long_chunks) > 1, f"前提：长文档应切分为多 chunk，实际 {len(long_chunks)}"
        total_before = r._bm25.doc_count

        # 按 doc_id 删除长文档
        r.remove_by_doc_ids(["d_sensetime_long"])

        # 验证：所有 chunk 都从 BM25 消失
        assert r._bm25.doc_count == total_before - len(long_chunks), (
            f"BM25 应减少 {len(long_chunks)} 条，"
            f"实际 {total_before} -> {r._bm25.doc_count}"
        )
        # 短文档仍在
        remaining_ids = {d["meta"].get("doc_id") for d in r.documents}
        assert "d_catl" in remaining_ids, "短文档不应被影响"
        assert "d_sensetime_long" not in remaining_ids, "长文档应完全移除"
        # BM25 可搜索性确认
        results = r._bm25.search(r.documents, "商汤科技营收", top_k=5)
        found_ids = {res["meta"]["doc_id"] for res in results}
        assert "d_sensetime_long" not in found_ids, "删除后仍能被 BM25 搜到"
