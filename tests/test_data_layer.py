"""
数据层可观测层测试 — 打开黑盒（纯加层，只读 introspection）

对应生产代码：financial_rag/services/data_layer.py + api/data_router.py

覆盖四条标准：
- 清单完整性：12 个存储全覆盖，layer/regenerable 语义正确，路径跟随 config.kb_dir
- 数据流图完整性：writes_to 全部指向清单内 id，关键操作齐全
- 实时巡检：计数反映磁盘实然，且严格只读（mtime 不变）
- 一致性体检：四方计数失配/缺失被标红，对齐时放行（critical）
"""
import json
import os
import sqlite3

import pytest

from financial_rag.config import config
from financial_rag.services import data_layer

pytestmark = pytest.mark.unit

# 计划中声明的全部存储（source 3 + index 5 + state 4）+ 补录（词典 + 输入区）
EXPECTED_STORE_IDS = {
    "kb_docs", "news_metadata", "news_archive",
    "kb_index", "bm25", "chroma", "embedding_cache", "lightrag",
    "stats", "kb_version", "conversations", "learning",
    "dictionaries", "financial_input",
}

REQUIRED_KEYS = {"id", "name", "layer", "owner", "writers", "regenerable", "desc"}


@pytest.fixture
def kb_tmp(tmp_path, monkeypatch):
    """把 KB 根目录指到临时目录，三层路径经 @property 自动跟随。

    补录存储同样隔离：data_dir（输入区）与词典目录（data_layer._DICT_DIR，
    源自 dictionary_registry 硬编码常量）都指到 tmp，不碰真实 data/。
    """
    root = tmp_path / "kb"
    monkeypatch.setattr(config, "kb_dir", str(root))
    monkeypatch.setattr(config, "data_dir", str(root / "financial"))
    monkeypatch.setattr(data_layer, "_DICT_DIR", root / "dictionaries")
    for d in (config.source_dir, config.index_dir, config.state_dir):
        os.makedirs(d, exist_ok=True)
    return root


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _write_jsonl(path, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"i": i}, ensure_ascii=False) + "\n")


def _make_bm25(db_path, n):
    """构造与 BM25Engine 同 schema 的 FTS5 库（只测巡检读取，不引引擎）。"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE VIRTUAL TABLE kb_fts USING fts5("
        "tokenized, doc_json UNINDEXED, doc_key UNINDEXED)"
    )
    conn.executemany(
        "INSERT INTO kb_fts(tokenized, doc_json, doc_key) VALUES (?,?,?)",
        [(f"tok{i}", "{}", f"d{i}") for i in range(n)],
    )
    conn.commit()
    conn.close()


def _make_chroma(chroma_dir, n):
    """构造含 embeddings 表的 chroma.sqlite3（只测巡检读取）。"""
    os.makedirs(chroma_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(chroma_dir, "chroma.sqlite3"))
    conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, embedding_id TEXT)")
    conn.executemany(
        "INSERT INTO embeddings(embedding_id) VALUES (?)",
        [(f"e{i}",) for i in range(n)],
    )
    conn.commit()
    conn.close()


def _seed_full_kb(root, doc_n=3, bm25_n=3, chroma_n=3):
    """铺一套四方对齐的完整数据（短文本 1 doc = 1 chunk）。"""
    src, idx, state = config.source_dir, config.index_dir, config.state_dir
    docs = [{"text": f"文档{i}", "meta": {"source": "test"}} for i in range(doc_n)]
    _write_json(os.path.join(src, "kb_docs.json"), docs)
    _write_json(os.path.join(idx, "kb_index.json"),
                {"version": 3, "doc_count": doc_n, "documents": docs, "config": {}})
    _make_bm25(os.path.join(idx, "bm25_index.db"), bm25_n)
    _make_chroma(os.path.join(idx, "chroma"), chroma_n)


# ===================== 清单完整性 =====================

class TestStoreRegistry:
    def test_covers_all_stores(self):
        ids = {s["id"] for s in data_layer.STORE_REGISTRY}
        assert ids == EXPECTED_STORE_IDS, f"清单缺/多存储: {ids ^ EXPECTED_STORE_IDS}"
        for s in data_layer.STORE_REGISTRY:
            assert REQUIRED_KEYS <= set(s), f"{s['id']} 缺字段: {REQUIRED_KEYS - set(s)}"

    def test_layer_and_regenerable_semantics(self):
        for s in data_layer.STORE_REGISTRY:
            assert s["layer"] in ("source", "index", "state", "input")
            if s["layer"] in ("source", "input"):
                assert s["regenerable"] is False, f"真相源/输入 {s['id']} 必须不可再生"
            else:
                assert s["regenerable"] is True, f"派生/状态 {s['id']} 必须可再生"

    def test_paths_follow_config_kb_dir(self, kb_tmp):
        m = data_layer.get_data_layer_map()
        for s in m["stores"]:
            if s.get("base", "kb") != "kb":
                continue  # 词典/输入区不在 kb_dir 下，另有路径源
            assert s["path"].startswith(str(kb_tmp)), \
                f"{s['id']} 路径未跟随 config.kb_dir: {s['path']}"


# ===================== 数据流图完整性 =====================

class TestFlowMap:
    def test_writes_to_reference_valid_stores(self):
        ids = {s["id"] for s in data_layer.STORE_REGISTRY}
        for flow in data_layer.FLOW_MAP:
            assert flow["writes_to"], f"{flow['id']} 落库列表为空"
            for target in flow["writes_to"]:
                assert target in ids, f"{flow['id']} 指向不存在的存储: {target}"

    def test_key_operations_present(self):
        flow_ids = {f["id"] for f in data_layer.FLOW_MAP}
        required = {"ingest_files", "ingest_news", "save_analysis",
                    "build_rebuild", "delete_docs", "clear_kb", "chat_followup"}
        assert required <= flow_ids, f"缺关键操作: {required - flow_ids}"


class TestDependencies:
    """上游依赖边（读依赖，非写入）：词典 → 分词/元数据抽取。"""

    def test_endpoints_reference_valid_stores(self):
        ids = {s["id"] for s in data_layer.STORE_REGISTRY}
        for dep in data_layer.DEPENDENCIES:
            assert dep["from"] in ids and dep["to"] in ids
            assert dep["note"]

    def test_dictionary_bm25_edge_present(self):
        edges = {(d["from"], d["to"]) for d in data_layer.DEPENDENCIES}
        assert ("dictionaries", "bm25") in edges, "缺词典→BM25 分词同源依赖边"


# ===================== 元数据字段覆盖率 =====================

class TestMetadataCoverage:
    @pytest.mark.critical
    def test_field_fill_counts(self, kb_tmp):
        docs = [
            {"text": "a", "meta": {"doc_id": "1", "source": "s",
                                    "company": "商汤科技", "sector": "AI"}},
            {"text": "b", "meta": {"doc_id": "2", "source": "s",
                                    "company": "宁德时代"}},
            {"text": "c", "meta": {"doc_id": "3", "source": "s"}},
        ]
        _write_json(os.path.join(config.source_dir, "kb_docs.json"), docs)
        cov = data_layer.metadata_coverage()
        assert cov["total"] == 3
        assert cov["fields"]["company"]["filled"] == 2
        assert cov["fields"]["company"]["pct"] == 66.7
        assert cov["fields"]["sector"]["filled"] == 1
        assert cov["fields"]["publish_date"]["filled"] == 0
        assert cov["fields"]["doc_id"]["filled"] == 3

    def test_empty_kb_no_crash(self, kb_tmp):
        cov = data_layer.metadata_coverage()
        assert cov["total"] == 0
        assert all(v["filled"] == 0 for v in cov["fields"].values())


# ===================== 实时巡检（只读） =====================

class TestInspectLiveState:
    def test_counts_reflect_disk(self, kb_tmp):
        src, idx, state = config.source_dir, config.index_dir, config.state_dir
        _seed_full_kb(kb_tmp, doc_n=3, bm25_n=5, chroma_n=7)
        _write_json(os.path.join(src, "news_metadata.json"), [{"t": 1}, {"t": 2}])
        _write_jsonl(os.path.join(src, "news_archive.jsonl"), 4)
        _write_jsonl(os.path.join(state, "learning", "history.jsonl"), 2)
        _write_json(os.path.join(idx, "embedding_cache.json"),
                    {"version": 1, "cache": {f"k{i}": [0.0] for i in range(6)}})
        _write_json(os.path.join(state, "stats.json"), {"kb_doc_count": 3})
        with open(os.path.join(state, ".kb_version"), "w") as f:
            f.write("7")
        conv_dir = os.path.join(state, "conversations")
        os.makedirs(conv_dir, exist_ok=True)
        for i in range(2):
            _write_json(os.path.join(conv_dir, f"s{i}.json"), {"id": f"s{i}"})

        live = data_layer.inspect_live_state()
        assert live["kb_docs"]["count"] == 3
        assert live["news_metadata"]["count"] == 2
        assert live["news_archive"]["count"] == 4
        assert live["kb_index"]["count"] == 3
        assert live["bm25"]["count"] == 5
        assert live["chroma"]["count"] == 7
        assert live["embedding_cache"]["count"] == 6
        assert live["learning"]["count"] == 2
        assert live["conversations"]["count"] == 2
        assert live["kb_version"]["exists"] is True
        assert live["stats"]["exists"] is True
        for sid in ("kb_docs", "bm25", "chroma"):
            assert live[sid]["size_kb"] > 0
            assert live[sid]["mtime"]

    def test_extended_stores_counts(self, kb_tmp):
        """补录存储：词典数 *.json，输入区数文件；路径分别跟随各自配置源。"""
        dict_dir = kb_tmp / "dictionaries"
        dict_dir.mkdir(parents=True)
        _write_json(str(dict_dir / "a.json"), {"jieba_words": ["算力"]})
        _write_json(str(dict_dir / "b.json"), {"stocks": {}})
        fin_dir = kb_tmp / "financial"
        fin_dir.mkdir(parents=True)
        (fin_dir / "doc.txt").write_text("x", encoding="utf-8")

        m = data_layer.get_data_layer_map()
        by_id = {s["id"]: s for s in m["stores"]}
        assert by_id["dictionaries"]["live"]["count"] == 2
        assert by_id["dictionaries"]["path"] == str(dict_dir)
        assert by_id["financial_input"]["live"]["count"] == 1
        assert by_id["financial_input"]["path"] == str(fin_dir)
        assert by_id["financial_input"]["layer"] == "input"

    def test_missing_stores_reported_not_crash(self, kb_tmp):
        live = data_layer.inspect_live_state()
        assert set(live) == EXPECTED_STORE_IDS
        for sid, info in live.items():
            assert info["exists"] is False, f"{sid} 空目录不应存在"
            assert info["count"] is None

    def test_inspection_is_readonly(self, kb_tmp):
        _seed_full_kb(kb_tmp)
        src, idx = config.source_dir, config.index_dir
        watched = [
            os.path.join(src, "kb_docs.json"),
            os.path.join(idx, "kb_index.json"),
            os.path.join(idx, "bm25_index.db"),
            os.path.join(idx, "chroma", "chroma.sqlite3"),
        ]
        before = {p: os.path.getmtime(p) for p in watched}
        data_layer.get_data_layer_map()
        data_layer.inspect_live_state()
        after = {p: os.path.getmtime(p) for p in watched}
        assert before == after, "巡检改写了数据文件（必须严格只读）"


# ===================== 一致性体检 =====================

class TestConsistencyCheck:
    @pytest.mark.critical
    def test_aligned_counts_pass(self, kb_tmp):
        _seed_full_kb(kb_tmp, doc_n=3, bm25_n=3, chroma_n=3)
        result = data_layer.consistency_check()
        assert result["ok"] is True
        assert result["issues"] == []
        assert result["counts"]["kb_docs"] == 3

    @pytest.mark.critical
    def test_mismatch_flagged(self, kb_tmp):
        _seed_full_kb(kb_tmp, doc_n=3, bm25_n=5, chroma_n=3)
        result = data_layer.consistency_check()
        assert result["ok"] is False
        assert any("bm25" in issue for issue in result["issues"]), \
            f"失配未被标红: {result['issues']}"

    @pytest.mark.critical
    def test_missing_derived_index_flagged(self, kb_tmp):
        _write_json(os.path.join(config.source_dir, "kb_docs.json"),
                    [{"text": "x", "meta": {}}])
        result = data_layer.consistency_check()
        assert result["ok"] is False
        assert any("bm25" in issue for issue in result["issues"])

    def test_empty_kb_is_consistent(self, kb_tmp):
        result = data_layer.consistency_check()
        assert result["ok"] is True


# ===================== API 端点 =====================

class TestDataLayerEndpoint:
    def test_get_data_layer_returns_map(self, kb_tmp):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from financial_rag.api.data_router import router

        _seed_full_kb(kb_tmp)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/api/data-layer")
        assert resp.status_code == 200
        body = resp.json()
        assert {"layers", "stores", "flows", "consistency",
                "dependencies", "metadata_coverage"} <= set(body)
        assert {s["id"] for s in body["stores"]} == EXPECTED_STORE_IDS
        assert body["consistency"]["ok"] is True
        assert body["metadata_coverage"]["total"] == 3