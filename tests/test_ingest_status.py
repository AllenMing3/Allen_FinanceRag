"""
数据导入状态可观测性 — 文件级导入状态与逐文件导入结果。

用户痛点：目录浏览器看不出"哪些文件已导入、哪些没导入"，导入后只有一个总数 toast。
本资产锁定两个纯函数的契约：
  ① _build_kb_file_index + _file_ingest_info：目录浏览器的文件级状态（已导入/待分析/未导入）
  ② _build_file_results：一次导入的逐文件结果（新增/重复/质量拒绝/解析为空）

对应生产代码：financial_rag/api/ingest_router.py
"""
import os

from financial_rag.api.ingest_router import (
    _build_kb_file_index,
    _file_ingest_info,
    _build_file_results,
)


def _doc(text="正文", source="a.txt", file=None, analyzed=False):
    meta = {"source": source, "analyzed": analyzed}
    if file is not None:
        meta["file"] = file
    return {"text": text, "meta": meta}


# ── ① 目录浏览器文件级状态 ──


class TestFileIngestStatus:
    def test_unimported_file(self):
        by_file, by_source = _build_kb_file_index([])
        info = _file_ingest_info("new.txt", os.path.join("data", "financial", "new.txt"), by_file, by_source)
        assert info == {"imported": False, "doc_count": 0, "analyzed_count": 0}

    def test_imported_with_analysis(self):
        docs = [_doc(file="data/financial/a.txt", analyzed=True),
                _doc(file="data/financial/a.txt", analyzed=True)]
        by_file, by_source = _build_kb_file_index(docs)
        info = _file_ingest_info("a.txt", os.path.join("data", "financial", "a.txt"), by_file, by_source)
        assert info["imported"] is True
        assert info["doc_count"] == 2
        assert info["analyzed_count"] == 2

    def test_imported_pending_analysis(self):
        docs = [_doc(file="data/financial/a.txt", analyzed=True),
                _doc(file="data/financial/a.txt", analyzed=False)]
        by_file, by_source = _build_kb_file_index(docs)
        info = _file_ingest_info("a.txt", os.path.join("data", "financial", "a.txt"), by_file, by_source)
        assert info["imported"] is True
        assert info["doc_count"] == 2
        assert info["analyzed_count"] == 1  # 前端据此显示"待分析"

    def test_path_normalization_matches(self):
        # KB 里存的是 ./data/financial\a.txt（未规范化），目录扫描给的是规范化路径
        docs = [_doc(file="./data/financial\\a.txt")]
        by_file, by_source = _build_kb_file_index(docs)
        info = _file_ingest_info("a.txt", os.path.normpath("./data/financial/a.txt"), by_file, by_source)
        assert info["imported"] is True

    def test_source_fallback_for_json_without_file(self):
        # .json 入库只带 source=fname，没有 meta.file → 按 source 兜底匹配
        docs = [_doc(source="reports.json"), _doc(source="reports.json")]
        by_file, by_source = _build_kb_file_index(docs)
        info = _file_ingest_info("reports.json", os.path.join("data", "x", "reports.json"), by_file, by_source)
        assert info["imported"] is True
        assert info["doc_count"] == 2

    def test_jsonl_custom_source_matched_by_file_not_source(self):
        # jsonl 的 source 可能是原始新闻源（≠文件名），必须靠 meta.file 精确归属，
        # 不能把别的文件的同名 source 文档算到本文件头上
        docs = [_doc(source="新浪财经", file="data/financial/news.jsonl"),
                _doc(source="新浪财经", file="data/financial/other.jsonl")]
        by_file, by_source = _build_kb_file_index(docs)
        info = _file_ingest_info("news.jsonl", os.path.join("data", "financial", "news.jsonl"), by_file, by_source)
        assert info["doc_count"] == 1

    def test_news_docs_do_not_pollute_file_status(self):
        # 新闻自动入库的 source="news:标题"，不应匹配任何目录文件
        docs = [_doc(source="news:某标题")]
        by_file, by_source = _build_kb_file_index(docs)
        info = _file_ingest_info("某标题.txt", os.path.join("data", "financial", "某标题.txt"), by_file, by_source)
        assert info["imported"] is False


# ── ② 逐文件导入结果 ──


class TestFileResults:
    def test_mixed_batch_attribution(self):
        fp_a = os.path.join("data", "x", "a.txt")
        fp_b = os.path.join("data", "x", "b.jsonl")
        # a.txt：读 2 → 清洗后 2 → 新增 1（另 1 重复）
        # b.jsonl：读 3 → 清洗后 1（2 篇被质量门拒绝）→ 新增 1
        cleaned = [_doc(file=fp_a), _doc(file=fp_a),
                   _doc(source="新浪", file=fp_b)]
        new_docs = [cleaned[0], cleaned[2]]
        entries = [("a.txt", fp_a, 2), ("b.jsonl", fp_b, 3)]
        results = _build_file_results(entries, cleaned, new_docs)
        ra, rb = results
        assert ra == {"file": "a.txt", "read": 2, "loaded": 1, "duplicates": 1, "rejected": 0, "status": "imported"}
        assert rb == {"file": "b.jsonl", "read": 3, "loaded": 1, "duplicates": 0, "rejected": 2, "status": "imported"}

    def test_all_duplicate_status(self):
        fp = os.path.join("data", "x", "a.txt")
        cleaned = [_doc(file=fp)]
        results = _build_file_results([("a.txt", fp, 1)], cleaned, [])
        assert results[0]["status"] == "duplicate"
        assert results[0]["duplicates"] == 1
        assert results[0]["loaded"] == 0

    def test_all_rejected_status(self):
        fp = os.path.join("data", "x", "a.txt")
        results = _build_file_results([("a.txt", fp, 3)], [], [])
        assert results[0]["status"] == "rejected"
        assert results[0]["rejected"] == 3

    def test_empty_parse_status(self):
        fp = os.path.join("data", "x", "scan.pdf")
        results = _build_file_results([("scan.pdf", fp, 0)], [], [])
        assert results[0]["status"] == "empty"
        assert results[0]["read"] == 0

    def test_json_source_attribution(self):
        # .json 入库无 meta.file，按 source==fname 归属
        fp = os.path.join("data", "x", "reports.json")
        cleaned = [_doc(source="reports.json")]
        results = _build_file_results([("reports.json", fp, 1)], cleaned, cleaned)
        assert results[0]["status"] == "imported"
        assert results[0]["loaded"] == 1

    def test_counts_conservation(self):
        # 守恒：read = loaded + duplicates + rejected（每个文件）
        fp = os.path.join("data", "x", "a.txt")
        cleaned = [_doc(file=fp)] * 5
        new_docs = cleaned[:2]
        r = _build_file_results([("a.txt", fp, 8)], cleaned, new_docs)[0]
        assert r["read"] == r["loaded"] + r["duplicates"] + r["rejected"]