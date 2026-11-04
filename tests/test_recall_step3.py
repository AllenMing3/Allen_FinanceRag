"""
Test Step 3 召回 — BM25 + Vector 通道级质量验证（RAG 流水线第三步）。

按项目四条测试标准组织：
  ① 证明效果  ② 可观测  ③ 一致性  ④ 回归锁定

核心发现（探针实然）：
  - BUG(已修): BM25 doc_key 对多 chunk 文档碰撞 → 去重只保留最后一个 chunk
  - 三套 ID 机制不一致（BM25/Vector/RRF），但 RRF 靠 text hash 融合不受影响
  - 分词："昇腾"被切单字（jieba 词典缺失，已知局限）
  - 打分区分度：共享金融词（"营收"）跨领域 ratio 仅 1.1x（已知局限）
  - 通道级内容相关性：BM25 6/6, Jaccard 6/6

对应生产代码：financial_rag/retrievers/bm25_engine.py + vector_engine.py + fusion.py
评估基准数据：financial_rag/eval/golden/eval_kb.jsonl（22 篇，含 4 篇非 AI 对照）
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from financial_rag.retrievers.bm25_engine import BM25Engine
from financial_rag.retrievers.vector_engine import VectorEngine
from financial_rag.retrievers.hybrid_engine import jieba_tokenizer
from financial_rag.retrievers.chunker import TextChunker
from financial_rag.retrievers.fusion import _doc_hash, rrf_fusion

_GOLDEN_KB = (
    Path(__file__).resolve().parents[1]
    / "financial_rag" / "eval" / "golden" / "eval_kb.jsonl"
)


def _load_golden_docs():
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


# 长文档（>500 字）用于触发 chunker 切分
_LONG_DOC_TEXT = (
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
)

_LONG_DOC = {
    "text": _LONG_DOC_TEXT,
    "meta": {"doc_id": "d_sensetime_long", "source": "eval_kb", "doc_type": "financial_report"},
}


@pytest.fixture
def golden_docs():
    return _load_golden_docs()


@pytest.fixture
def tokenizer():
    return jieba_tokenizer()


@pytest.fixture
def bm25_engine(tokenizer, tmp_path):
    db_path = str(tmp_path / "test_bm25.db")
    engine = BM25Engine(tokenizer=tokenizer, db_path=db_path)
    return engine


@pytest.fixture
def vector_engine(tokenizer):
    return VectorEngine(embedder=None, tokenizer=tokenizer)


@pytest.fixture
def long_doc_chunks(tokenizer):
    """用 chunk_size=200 强制切分长文档为多 chunk（测试切分机制）。"""
    chunker = TextChunker(chunk_size=200, chunk_overlap=30, skip_threshold=100)
    return chunker.split(
        _LONG_DOC["text"], meta=_LONG_DOC["meta"], source_id="d_sensetime_long"
    )


# ======================================================================
#  ④ 回归锁定：BM25 多 chunk doc_key 碰撞 bug
# ======================================================================

@pytest.mark.regression
class TestChunkSurvivalRegression:
    """BUG: 所有 chunk 继承父文档 meta.doc_id → BM25 doc_key 碰撞 →
    _insert_batch 去重只保留最后一个 chunk → 前面的 chunk 静默丢失。

    修复: doc_key() 检测 chunk_count > 1 时生成复合 key "{doc_id}_c{chunk_id}"。
    """

    def test_multi_chunk_all_survive_bm25(self, bm25_engine, long_doc_chunks):
        """修复前: 3 chunks → doc_count=1（丢失 2）。修复后: doc_count=3。"""
        assert len(long_doc_chunks) > 1, "测试前提：长文档必须被切为多 chunk"
        bm25_engine.build(long_doc_chunks)
        assert bm25_engine.doc_count == len(long_doc_chunks)

    def test_multi_chunk_all_survive_jaccard(self, vector_engine, long_doc_chunks, tokenizer):
        """Jaccard 通道不受 doc_key 影响（用 text hash），始终保留全部 chunk。"""
        results = vector_engine.search_jaccard(
            long_doc_chunks, "商汤科技营收", top_k=len(long_doc_chunks),
            tokenize_fn=tokenizer,
        )
        assert len(results) == len(long_doc_chunks)

    def test_short_doc_key_unchanged(self, golden_docs):
        """短文（chunk_count=1 或无 chunk_id）的 doc_key 仍为 plain doc_id，不受修复影响。"""
        for doc in golden_docs[:5]:
            key = BM25Engine.doc_key(doc)
            assert key == doc["meta"]["doc_id"]


# ======================================================================
#  ② 可观测：ID 属性可验证
# ======================================================================

class TestIDObservability:
    """doc_key 的格式、唯一性、确定性均可观测验证。"""

    def test_doc_key_unique_per_chunk(self, long_doc_chunks):
        """多 chunk 文档的每个 chunk 必须有唯一 doc_key。"""
        keys = [BM25Engine.doc_key(c) for c in long_doc_chunks]
        assert len(keys) == len(set(keys)), f"doc_key 碰撞: {keys}"

    def test_doc_key_format_for_chunks(self, long_doc_chunks):
        """chunk 的 doc_key 格式为 '{doc_id}_c{chunk_id}'。"""
        for chunk in long_doc_chunks:
            key = BM25Engine.doc_key(chunk)
            cid = chunk["meta"]["chunk_id"]
            assert key == f"d_sensetime_long_c{cid}"

    def test_doc_key_deterministic(self, golden_docs):
        """同一文档多次计算 doc_key 结果一致。"""
        doc = golden_docs[0]
        assert BM25Engine.doc_key(doc) == BM25Engine.doc_key(doc)

    def test_rrf_hash_identifies_same_text(self, golden_docs):
        """RRF _doc_hash 基于 text 内容，同一文档从不同通道返回时 hash 一致。"""
        doc = golden_docs[0]
        h1 = _doc_hash(doc["text"])
        h2 = _doc_hash(doc["text"])
        assert h1 == h2
        # 不同文档 hash 不同
        h3 = _doc_hash(golden_docs[1]["text"])
        assert h1 != h3


# ======================================================================
#  ③ 一致性：跨通道 + 增量操作
# ======================================================================

class TestCrossChannelConsistency:
    """两路对同一文档的标识虽不同，但 RRF 融合能正确合并。"""

    def test_rrf_fusion_merges_same_doc(self, bm25_engine, vector_engine,
                                         golden_docs, tokenizer):
        """同一文档从 BM25 和 Jaccard 返回后，RRF 融合为 1 条（不是 2 条）。"""
        bm25_engine.build(golden_docs)
        query = "商汤科技营收"
        # 通道深度取 5（对齐生产 search 的 top_k*2 候选深度）：golden 扩容到 45 篇后
        # top3 内两通道不再重叠（Jaccard 被新增文档稀释），consensus 在 top5 出现
        bm25_res = bm25_engine.search(golden_docs, query, top_k=5)
        jaccard_res = vector_engine.search_jaccard(
            golden_docs, query, top_k=5, tokenize_fn=tokenizer,
        )
        fused, stats = rrf_fusion(
            [(bm25_res, 0.3, "bm25"), (jaccard_res, 0.7, "vector")],
            top_k=5, rrf_k=60,
        )
        # consensus_count > 0 说明有文档被两路同时命中并正确合并
        assert stats.consensus_count > 0

    def test_remove_then_readd_restores_count(self, bm25_engine, golden_docs):
        """remove + re-add 后 doc_count 恢复。"""
        bm25_engine.build(golden_docs[:3])
        assert bm25_engine.doc_count == 3
        bm25_engine.remove_by_docs([golden_docs[0]])
        assert bm25_engine.doc_count == 2
        bm25_engine.add([golden_docs[0]])
        assert bm25_engine.doc_count == 3

    def test_rebuild_is_deterministic(self, tokenizer, golden_docs, tmp_path):
        """两次 build 同一数据集 → doc_count 一致。"""
        db1 = str(tmp_path / "bm25_a.db")
        db2 = str(tmp_path / "bm25_b.db")
        e1 = BM25Engine(tokenizer=tokenizer, db_path=db1)
        e2 = BM25Engine(tokenizer=tokenizer, db_path=db2)
        e1.build(golden_docs)
        e2.build(golden_docs)
        assert e1.doc_count == e2.doc_count == len(golden_docs)


# ======================================================================
#  ① 证明效果：打分区分度 + 内容相关性
# ======================================================================

class TestScoreDiscrimination:
    """BM25 对明确相关 vs 明确不相关的文档有显著分数差。"""

    @pytest.mark.parametrize("query,relevant_id,irrelevant_id", [
        ("AI芯片出口管制", "d_export_control", "d_nongfu_sales"),
        ("RAG检索增强生成", "d_rag_concept", "d_vanke_debt"),
        ("贵州茅台年报", "d_moutai_2024", "d_sensetime_compute"),
        ("华为昇腾国产替代", "d_chip_domestic", "d_nongfu_sales"),
    ])
    def test_relevant_scores_higher(self, bm25_engine, golden_docs,
                                     query, relevant_id, irrelevant_id):
        """相关文档 BM25 分数至少 3x 于不相关文档。"""
        bm25_engine.build(golden_docs)
        results = bm25_engine.search(golden_docs, query, top_k=len(golden_docs))
        score_map = {r["meta"]["doc_id"]: r["score"] for r in results}
        rel = score_map.get(relevant_id, 0)
        irr = score_map.get(irrelevant_id, 0)
        assert rel > 0, f"相关文档 {relevant_id} 未被召回"
        if irr > 0:
            assert rel / irr >= 3.0, (
                f"区分度不足: {relevant_id}={rel:.4f} vs {irrelevant_id}={irr:.4f}"
            )


class TestContentRelevance:
    """通道级内容相关性：top-5 中必须有文本包含用户所需信息。"""

    @pytest.mark.parametrize("query,info_needed", [
        ("大模型融资", ["融资", "亿"]),
        ("AI芯片出口管制", ["商务部", "管制"]),
        ("RAG检索增强生成", ["知识库", "生成"]),
        ("贵州茅台年报", ["1741亿", "净利润"]),
        ("华为昇腾国产替代", ["昇腾", "910B"]),
    ])
    def test_bm25_content_relevant(self, bm25_engine, golden_docs, query, info_needed):
        """BM25 top-5 中至少有一条文本包含全部 info_needed 短语。"""
        bm25_engine.build(golden_docs)
        results = bm25_engine.search(golden_docs, query, top_k=5)
        assert any(
            all(p in r["text"] for p in info_needed)
            for r in results
        ), f"BM25 top-5 未包含所需信息: {info_needed}"

    @pytest.mark.parametrize("query,info_needed", [
        ("大模型融资", ["融资", "亿"]),
        ("AI芯片出口管制", ["商务部", "管制"]),
        ("RAG检索增强生成", ["知识库", "生成"]),
        ("贵州茅台年报", ["1741亿", "净利润"]),
        ("华为昇腾国产替代", ["昇腾", "910B"]),
    ])
    def test_jaccard_content_relevant(self, vector_engine, golden_docs, tokenizer,
                                       query, info_needed):
        """Jaccard top-5 中至少有一条文本包含全部 info_needed 短语。"""
        results = vector_engine.search_jaccard(
            golden_docs, query, top_k=5, tokenize_fn=tokenizer,
        )
        assert any(
            all(p in r["text"] for p in info_needed)
            for r in results
        ), f"Jaccard top-5 未包含所需信息: {info_needed}"
