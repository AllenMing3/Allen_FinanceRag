"""
Test persistence layer — doc_id, dedup, backup, rotation, index persistence

Covers:
- make_doc_id: stable hash generation
- assign_doc_ids: idempotent assignment
- dedup_docs: filter duplicates against existing
- _atomic_write_json: auto-backup (.bak)
- _rotate_jsonl: news archive cap enforcement
- Index save/load: HybridRetriever round-trip
"""
import json
import os
import tempfile
import pytest

from financial_rag.services.persistence import (
    make_doc_id, assign_doc_ids, dedup_docs,
    _atomic_write_json, _rotate_jsonl,
    load_kb, save_kb, load_meta, save_meta,
    append_news_archive,
)


# ===================== Doc ID =====================


class TestMakeDocId:

    def test_stable_hash(self):
        """Same content → same ID"""
        d1 = {"text": "商汤科技营收50亿", "meta": {"source": "news.jsonl"}}
        d2 = {"text": "商汤科技营收50亿", "meta": {"source": "news.jsonl"}}
        assert make_doc_id(d1) == make_doc_id(d2)

    def test_different_text(self):
        d1 = {"text": "商汤科技营收50亿", "meta": {"source": "news.jsonl"}}
        d2 = {"text": "英伟达发布B200", "meta": {"source": "news.jsonl"}}
        assert make_doc_id(d1) != make_doc_id(d2)

    def test_different_source(self):
        d1 = {"text": "same text", "meta": {"source": "file1.txt"}}
        d2 = {"text": "same text", "meta": {"source": "file2.txt"}}
        assert make_doc_id(d1) != make_doc_id(d2)

    def test_truncated_text(self):
        """Only first 200 chars matter for ID"""
        base = "A" * 200
        d1 = {"text": base + "EXTRA1", "meta": {"source": "s"}}
        d2 = {"text": base + "EXTRA2", "meta": {"source": "s"}}
        assert make_doc_id(d1) == make_doc_id(d2)

    def test_empty_meta(self):
        d = {"text": "hello", "meta": {}}
        did = make_doc_id(d)
        assert len(did) == 16  # SHA-256 truncated to 16 hex

    def test_no_meta(self):
        d = {"text": "hello"}
        did = make_doc_id(d)
        assert len(did) == 16


class TestAssignDocIds:

    def test_assigns_ids(self):
        docs = [
            {"text": "doc A", "meta": {"source": "s1"}},
            {"text": "doc B", "meta": {"source": "s2"}},
        ]
        assign_doc_ids(docs)
        assert docs[0]["meta"]["doc_id"] is not None
        assert docs[1]["meta"]["doc_id"] is not None
        assert docs[0]["meta"]["doc_id"] != docs[1]["meta"]["doc_id"]

    def test_idempotent(self):
        """Re-assigning doesn't change existing IDs"""
        docs = [{"text": "doc A", "meta": {"source": "s1"}}]
        assign_doc_ids(docs)
        original_id = docs[0]["meta"]["doc_id"]
        assign_doc_ids(docs)
        assert docs[0]["meta"]["doc_id"] == original_id

    def test_creates_meta_if_missing(self):
        docs = [{"text": "no meta"}]
        assign_doc_ids(docs)
        assert "meta" in docs[0]
        assert "doc_id" in docs[0]["meta"]


# ===================== Dedup =====================


class TestDedupDocs:

    def test_filters_duplicates(self):
        existing = [
            {"text": "doc A", "meta": {"source": "s1", "doc_id": "aaa"}},
        ]
        new_docs = [
            {"text": "doc A", "meta": {"source": "s1", "doc_id": "aaa"}},
            {"text": "doc B", "meta": {"source": "s2", "doc_id": "bbb"}},
        ]
        result = dedup_docs(existing, new_docs)
        assert len(result) == 1
        assert result[0]["meta"]["doc_id"] == "bbb"

    def test_all_new(self):
        existing = []
        new_docs = [
            {"text": "A", "meta": {"doc_id": "x"}},
            {"text": "B", "meta": {"doc_id": "y"}},
        ]
        result = dedup_docs(existing, new_docs)
        assert len(result) == 2

    def test_all_duplicate(self):
        existing = [{"text": "A", "meta": {"doc_id": "x"}}]
        new_docs = [{"text": "A", "meta": {"doc_id": "x"}}]
        result = dedup_docs(existing, new_docs)
        assert len(result) == 0

    def test_dedup_within_batch(self):
        """Duplicates within the new batch itself are also filtered"""
        existing = []
        new_docs = [
            {"text": "A", "meta": {"doc_id": "x"}},
            {"text": "A", "meta": {"doc_id": "x"}},  # dup within batch
        ]
        result = dedup_docs(existing, new_docs)
        assert len(result) == 1


# ===================== Backup =====================


class TestAtomicWriteBackup:

    def test_creates_backup(self, tmp_path):
        path = str(tmp_path / "test.json")
        # First write — no backup needed
        _atomic_write_json(path, {"v": 1})
        assert not os.path.exists(path + ".bak")

        # Second write — creates .bak
        _atomic_write_json(path, {"v": 2})
        assert os.path.exists(path + ".bak")

        # .bak contains previous content
        with open(path + ".bak", "r") as f:
            bak = json.load(f)
        assert bak["v"] == 1

        # Current file has new content
        with open(path, "r") as f:
            current = json.load(f)
        assert current["v"] == 2

    def test_backup_disabled(self, tmp_path):
        path = str(tmp_path / "test.json")
        _atomic_write_json(path, {"v": 1})
        _atomic_write_json(path, {"v": 2}, backup=False)
        assert not os.path.exists(path + ".bak")


# ===================== Rotation =====================


class TestRotateJsonl:

    def test_no_rotation_under_limit(self, tmp_path):
        path = str(tmp_path / "archive.jsonl")
        with open(path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"line": i}) + "\n")
        _rotate_jsonl(path, 20)
        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 10

    def test_rotation_trims_oldest(self, tmp_path):
        path = str(tmp_path / "archive.jsonl")
        with open(path, "w") as f:
            for i in range(100):
                f.write(json.dumps({"line": i}) + "\n")
        _rotate_jsonl(path, 50)
        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 50
        # First line should be line 50 (kept newest)
        first = json.loads(lines[0])
        assert first["line"] == 50

    def test_rotation_nonexistent_file(self, tmp_path):
        """No error on missing file"""
        _rotate_jsonl(str(tmp_path / "missing.jsonl"), 100)

    def test_news_archive_rotation(self, tmp_path):
        path = str(tmp_path / "archive.jsonl")
        items = [{"title": f"News {i}", "content": f"Content {i}"} for i in range(100)]
        # Append with low max_lines
        append_news_archive(items, "test", path=path, max_lines=30)
        with open(path, "r") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 30


# ===================== Index Persistence =====================


class TestIndexPersistence:

    def test_save_load_roundtrip(self, tmp_path):
        """Save index → load → same documents"""
        from financial_rag.retrievers.hybrid_engine import HybridRetriever

        docs = [
            {"text": "商汤科技2024年营收增长36%", "meta": {"source": "test1"}},
            {"text": "英伟达Blackwell架构GPU发布", "meta": {"source": "test2"}},
            {"text": "OpenAI GPT-5即将发布", "meta": {"source": "test3"}},
        ]

        r1 = HybridRetriever()
        r1.index(docs, precompute_embeddings=False)  # No embedding (no API key)

        index_path = str(tmp_path / "index.json")
        r1.save_index(index_path)
        assert os.path.exists(index_path)

        # Load into new retriever
        r2 = HybridRetriever()
        r2.load_index(index_path)
        assert len(r2.documents) == 3

        # Search results match
        results1 = r1.search("商汤科技", top_k=3, use_rerank=False)
        results2 = r2.search("商汤科技", top_k=3, use_rerank=False)
        assert len(results1) == len(results2)
        assert results1[0]["text"] == results2[0]["text"]

    def test_kb_save_load_roundtrip(self, tmp_path):
        """KB docs save/load preserves content"""
        path = str(tmp_path / "kb.json")
        docs = [
            {"text": "doc1", "meta": {"source": "s1", "doc_id": "abc"}},
            {"text": "doc2", "meta": {"source": "s2", "doc_id": "def"}},
        ]
        save_kb(docs, path=path)
        loaded = load_kb(path=path)
        assert len(loaded) == 2
        assert loaded[0]["meta"]["doc_id"] == "abc"

    def test_meta_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "meta.json")
        meta = [{"keyword": "AI", "title": "test"}]
        save_meta(meta, path=path)
        loaded = load_meta(path=path)
        assert len(loaded) == 1
        assert loaded[0]["keyword"] == "AI"
