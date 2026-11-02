"""
L1 golden 语料库完整性自检 — 把 Step 3 教训制度化为测试护栏。

Step 3 血泪教训：info_needed 靠假设（把商汤营收猜成 "29.61亿/176%"，
实际是 "50.3亿/36.4%"），差点把"探针设计错误"误判为"检索失败"。

本文件钉死 golden 数据的自洽性，全部离线确定、无需检索器：
  1. schema 完整：每条 qrels 字段齐全
  2. doc_id 有效：qrels 引用的 doc_id 都在 eval_kb 中
  3. info_needed 真实：每条 query 的信息锚点逐字出现在其正例(grade>=1)正文
  4. 负例存在：每条 query 至少 1 个 grade=0 显式负例（错误 case）
  5. 无孤儿：eval_kb 每篇文档至少被一条 qrels 引用（正例或负例）
  6. 分级合法：grade ∈ {0,1,2,3}
  7. 长文可切：d_sensetime_long 经 news 策略切成 >=2 块（覆盖 doc_key 碰撞场景）

对应数据：financial_rag/eval/golden/{eval_kb,eval_kb_long,qrels}.jsonl
  - eval_kb.jsonl:      短文基准库（22 篇，均 <500 字，与 step1 入库测试共享契约）
  - eval_kb_long.jsonl: L1 长文场景专属（触发 chunker 切分，覆盖 doc_key 碰撞）
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from financial_rag.retrievers.chunker import TextChunker

_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "financial_rag" / "eval" / "golden"
_EVAL_KB = _GOLDEN_DIR / "eval_kb.jsonl"
_EVAL_KB_LONG = _GOLDEN_DIR / "eval_kb_long.jsonl"
_QRELS = _GOLDEN_DIR / "qrels.jsonl"

_REQUIRED_FIELDS = {"query_id", "query", "relevant_docs", "info_needed", "top_k", "tags"}


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def kb_docs():
    # 短文基准库 + 长文场景库合并为完整 golden 语料
    return _load_jsonl(_EVAL_KB) + _load_jsonl(_EVAL_KB_LONG)


@pytest.fixture(scope="module")
def kb_text(kb_docs):
    """{doc_id: text}"""
    return {d["doc_id"]: d["text"] for d in kb_docs}


@pytest.fixture(scope="module")
def qrels():
    return _load_jsonl(_QRELS)


@pytest.mark.unit
class TestQrelsSchema:
    def test_query_ids_unique(self, qrels):
        ids = [q["query_id"] for q in qrels]
        assert len(ids) == len(set(ids)), f"query_id 重复: {[i for i, c in Counter(ids).items() if c > 1]}"

    def test_required_fields_present(self, qrels):
        for q in qrels:
            missing = _REQUIRED_FIELDS - set(q.keys())
            assert not missing, f"{q.get('query_id')} 缺字段: {missing}"

    def test_grades_in_range(self, qrels):
        for q in qrels:
            for doc_id, g in q["relevant_docs"].items():
                assert g in (0, 1, 2, 3), f"{q['query_id']} 的 {doc_id} grade={g} 非法（须 0-3）"

    def test_top_k_positive(self, qrels):
        for q in qrels:
            assert isinstance(q["top_k"], int) and q["top_k"] > 0, f"{q['query_id']} top_k 非法"


@pytest.mark.unit
class TestDocIdIntegrity:
    def test_all_doc_ids_exist_in_kb(self, qrels, kb_text):
        for q in qrels:
            for doc_id in q["relevant_docs"]:
                assert doc_id in kb_text, f"{q['query_id']} 引用了不存在的 doc_id: {doc_id}"

    def test_no_orphan_docs(self, qrels, kb_docs):
        """每篇 eval_kb 文档至少被一条 qrels 引用（正例或负例）——杜绝孤儿语料。"""
        referenced = {doc_id for q in qrels for doc_id in q["relevant_docs"]}
        all_ids = {d["doc_id"] for d in kb_docs}
        orphans = all_ids - referenced
        assert not orphans, f"孤儿文档（从未被任何 qrels 引用）: {sorted(orphans)}"


@pytest.mark.unit
class TestNegatives:
    """错误 case（grade=0 负例）— Step 3 '打分区分度' 维度的数据基础。"""

    def test_every_query_has_negative(self, qrels):
        for q in qrels:
            negs = [d for d, g in q["relevant_docs"].items() if g == 0]
            assert negs, f"{q['query_id']} 没有 grade=0 负例，无法验证打分区分度"

    def test_every_query_has_positive(self, qrels):
        for q in qrels:
            pos = [d for d, g in q["relevant_docs"].items() if g >= 1]
            assert pos, f"{q['query_id']} 没有 grade>=1 正例"

    def test_negative_disjoint_from_positive(self, qrels):
        """同一 query 内，一个 doc_id 不能既是正例又是负例（dict key 天然唯一，双保险）。"""
        for q in qrels:
            pos = {d for d, g in q["relevant_docs"].items() if g >= 1}
            neg = {d for d, g in q["relevant_docs"].items() if g == 0}
            assert pos.isdisjoint(neg), f"{q['query_id']} 正负例重叠: {pos & neg}"


@pytest.mark.unit
class TestInfoNeeded:
    """info_needed 真实性 — 直接钉死 Step 3 '靠假设编锚点' 的教训。"""

    def test_info_needed_appears_in_positive_docs(self, qrels, kb_text):
        for q in qrels:
            pos_ids = [d for d, g in q["relevant_docs"].items() if g >= 1]
            corpus = " ".join(kb_text[d] for d in pos_ids)
            for phrase in q["info_needed"]:
                assert phrase in corpus, (
                    f"{q['query_id']} 的 info_needed '{phrase}' 未出现在正例正文 "
                    f"{pos_ids}（疑似假设，非真实文本）"
                )

    def test_info_needed_non_empty(self, qrels):
        for q in qrels:
            assert q["info_needed"], f"{q['query_id']} info_needed 为空"


@pytest.mark.unit
class TestLongDocChunking:
    """长文切分 — 覆盖 BM25 doc_key 碰撞 bug 场景（Step 1 强化测试模式）。"""

    def test_sensetime_long_splits_into_multiple_chunks(self, kb_docs):
        long_doc = next(d for d in kb_docs if d["doc_id"] == "d_sensetime_long")
        # 生产 chunker: skip_threshold=500；news 策略 chunk_size=500
        chunker = TextChunker(chunk_size=1500, chunk_overlap=100,
                              min_chunk_size=80, skip_threshold=500)
        chunks = chunker.split_by_doctype(
            long_doc["text"], doc_type=long_doc["doc_type"],
            meta={"doc_id": long_doc["doc_id"], "doc_type": long_doc["doc_type"]},
            source_id=long_doc["doc_id"],
        )
        assert len(chunks) >= 2, (
            f"d_sensetime_long(doc_type={long_doc['doc_type']}, {len(long_doc['text'])}字) "
            f"只切成 {len(chunks)} 块，无法覆盖 doc_key 碰撞场景"
        )

    def test_chunks_share_doc_id_with_unique_chunk_id(self, kb_docs):
        long_doc = next(d for d in kb_docs if d["doc_id"] == "d_sensetime_long")
        chunker = TextChunker(skip_threshold=500)
        chunks = chunker.split_by_doctype(
            long_doc["text"], doc_type=long_doc["doc_type"],
            meta={"doc_id": long_doc["doc_id"], "doc_type": long_doc["doc_type"]},
            source_id=long_doc["doc_id"],
        )
        doc_ids = {c["meta"]["doc_id"] for c in chunks}
        chunk_ids = [c["meta"]["chunk_id"] for c in chunks]
        assert doc_ids == {"d_sensetime_long"}, "所有 chunk 必须共享同一逻辑 doc_id"
        assert len(chunk_ids) == len(set(chunk_ids)), f"chunk_id 必须唯一: {chunk_ids}"
