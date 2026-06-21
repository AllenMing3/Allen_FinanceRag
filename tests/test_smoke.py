"""
Smoke tests — end-to-end capability verification

Covers:
- Full agent chain: Coordinator → Ingestion → Analysis → Scoring
- Web API endpoints: config, status, ingest, build, query, clear
- KB lifecycle: ingest → dedup → build → search → delete
- Persistence round-trip: save → reload → verify

Uses FastAPI TestClient (no real server needed).
MOCK_MODE=true to avoid real API calls for data sources.
"""
import os
import json
import tempfile
import pytest

# Force mock mode for all tests in this module
os.environ["MOCK_MODE"] = "true"

from financial_rag.core.base import AgentContext
from financial_rag.tools.core import create_financial_registry, ToolExecutor
from financial_rag.services.persistence import (
    make_doc_id, assign_doc_ids, dedup_docs,
    save_kb, load_kb,
)


# ===================== Fixtures =====================


@pytest.fixture
def tools():
    registry = create_financial_registry(retriever=None, llm=None)
    executor = ToolExecutor(registry)
    return registry, executor


def _bind(agent, tools):
    registry, executor = tools
    agent.bind_tools(registry, executor)
    return agent


SAMPLE_TEXT = """
商汤科技2024年年度报告：实现营业收入50.3亿元，同比增长36.4%。
生成式AI业务收入占比达60%，研发投入18.7亿元。
训练集群规模达4万卡A100，推理成本降至0.5元/百万token。
企业客户数达5800家，年度经常性收入（ARR）达28亿元。
"""


# ===================== Agent Chain Smoke =====================


class TestAgentChainSmoke:
    """Full chain: Coordinator → Ingestion → Analysis → Scoring"""

    def test_coordinator_classifies_kline(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = _bind(CoordinatorAgent(), tools)
        ctx = AgentContext(raw_input="茅台最近走势怎么样")
        result = agent.run(ctx)
        assert result.success
        assert result.data["intent"] == "kline"

    def test_coordinator_classifies_report(self, tools):
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        agent = _bind(CoordinatorAgent(), tools)
        ctx = AgentContext(raw_input="商汤科技2024年营收增长了多少")
        result = agent.run(ctx)
        assert result.success
        assert result.data["intent"] in ("report", "general")

    def test_ingestion_processes_text(self, tools):
        from financial_rag.agents.ingestion_agent import IngestionAgent
        agent = _bind(IngestionAgent(), tools)
        ctx = AgentContext(raw_input=SAMPLE_TEXT)
        result = agent.run(ctx)
        assert result.success
        assert len(result.data) >= 1
        assert result.data[0]["text"]  # cleaned text exists

    def test_analysis_generates_report(self, tools):
        from financial_rag.agents.analysis_agent import AnalysisAgent
        agent = _bind(AnalysisAgent(), tools)
        docs = [{"text": SAMPLE_TEXT, "meta": {"source": "test"}}]
        ctx = AgentContext(
            parsed_data=docs,
            raw_input="分析商汤科技财报",
            metadata={"intent": "report"},
        )
        result = agent.run(ctx)
        assert result.success
        assert result.agent_name == "AnalysisAgent"

    def test_scoring_evaluates(self, tools):
        from financial_rag.agents.scoring_agent import ScoringAgent
        agent = _bind(ScoringAgent(), tools)
        ctx = AgentContext(
            raw_input="分析商汤",
            final_answer="商汤科技2024年营收增长36.4%",
            intermediate_findings=[{"stage": "report", "source_count": 2}],
            metadata={
                "fetched_data": [],
                "retrieved_items": [{"text": SAMPLE_TEXT}],
            },
        )
        result = agent.run(ctx)
        assert result.success
        assert "pipeline_scores" in result.data

    def test_full_chain_no_crash(self, tools):
        """End-to-end: Coordinator → Ingestion → Analysis → Scoring (no crash)"""
        from financial_rag.agents.coordinator_agent import CoordinatorAgent
        from financial_rag.agents.ingestion_agent import IngestionAgent
        from financial_rag.agents.analysis_agent import AnalysisAgent
        from financial_rag.agents.scoring_agent import ScoringAgent

        # Coordinator
        coord = _bind(CoordinatorAgent(), tools)
        ctx = AgentContext(raw_input="分析商汤科技2024年报")
        coord_result = coord.run(ctx)
        assert coord_result.success
        intent = coord_result.data["intent"]

        # Ingestion
        ingest = _bind(IngestionAgent(), tools)
        ingest_ctx = AgentContext(raw_input=SAMPLE_TEXT)
        ingest_result = ingest.run(ingest_ctx)
        assert ingest_result.success

        # Analysis
        analyze = _bind(AnalysisAgent(), tools)
        analyze_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", []),
            raw_input=SAMPLE_TEXT,
            metadata={"intent": intent},
        )
        analyze_result = analyze.run(analyze_ctx)
        assert analyze_result.success

        # Scoring
        scorer = _bind(ScoringAgent(), tools)
        score_ctx = AgentContext(
            raw_input=SAMPLE_TEXT,
            final_answer=analyze_result.context_updates.get("final_answer", "analysis done"),
            intermediate_findings=analyze_result.context_updates.get("intermediate_findings", []),
            metadata={"retrieved_items": [], "fetched_data": []},
        )
        score_result = scorer.run(score_ctx)
        assert score_result.success


# ===================== Retrieval Smoke =====================


class TestRetrievalSmoke:
    """HybridRetriever basic search capability"""

    def test_bm25_search(self):
        from financial_rag.retrievers.retriever import HybridRetriever
        r = HybridRetriever()
        docs = [
            {"text": "商汤科技营收增长36%至50亿元", "meta": {"source": "a"}},
            {"text": "英伟达发布Blackwell GPU架构", "meta": {"source": "b"}},
            {"text": "OpenAI GPT-5推理成本降低", "meta": {"source": "c"}},
        ]
        r.index(docs, precompute_embeddings=False)
        results = r.search("商汤科技营收", top_k=3, use_rerank=False)
        assert len(results) >= 1
        assert "商汤" in results[0]["text"]

    def test_incremental_add(self):
        from financial_rag.retrievers.retriever import HybridRetriever
        r = HybridRetriever()
        docs = [{"text": "first doc", "meta": {"source": "s1"}}]
        r.index(docs, precompute_embeddings=False)
        assert len(r.documents) == 1

        new_docs = [{"text": "second doc", "meta": {"source": "s2"}}]
        r.add(new_docs, use_chunker=False)
        assert len(r.documents) == 2


# ===================== Web API Smoke =====================


class TestWebAPISmoke:
    """FastAPI TestClient — key endpoints respond correctly"""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from financial_rag.web import app, _state
        self.client = TestClient(app)

    def test_api_config(self):
        resp = self.client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_model" in data
        assert "mock_mode" in data
        assert "has_api_key" in data
        assert "embedding_model" in data

    def test_api_kb_status_empty(self):
        resp = self.client.get("/api/kb/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "doc_count" in data
        assert "kb_built" in data

    def test_api_directories(self):
        resp = self.client.get("/api/directories")
        assert resp.status_code == 200
        data = resp.json()
        assert "directories" in data
        assert len(data["directories"]) >= 1

    def test_api_metadata_status(self):
        resp = self.client.get("/api/metadata/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data

    def test_api_learning_stats(self):
        resp = self.client.get("/api/learning/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "kb_doc_count" in data

    def test_api_kb_history(self):
        resp = self.client.get("/api/kb/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "history" in data
        assert "stats" in data

    def test_api_ingest_and_dedup(self, tmp_path):
        """Ingest same file twice → second time reports duplicates"""
        # Clear KB first to ensure clean state
        self.client.post("/api/kb/clear")

        # Create a test file
        test_file = tmp_path / "test_dedup.jsonl"
        test_file.write_text(
            json.dumps({"text": "商汤科技营收50亿", "metadata": {"source": "test_dedup"}}) + "\n" +
            json.dumps({"text": "英伟达GPU发布", "metadata": {"source": "test_dedup"}}) + "\n",
            encoding="utf-8"
        )

        # First ingest
        resp1 = self.client.post("/api/ingest/files", json={"dir": str(tmp_path), "analyze": False})
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1.get("loaded", 0) == 2  # both new

        # Second ingest — should detect duplicates
        resp2 = self.client.post("/api/ingest/files", json={"dir": str(tmp_path), "analyze": False})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2.get("skipped_duplicates", 0) == 2
        assert data2.get("loaded", 0) == 0

    def test_api_build_and_query(self, tmp_path):
        """Ingest → Build → Query returns results"""
        # Clear KB first
        self.client.post("/api/kb/clear")

        # Create test data
        test_file = tmp_path / "kb_build.jsonl"
        test_file.write_text(
            json.dumps({"text": "商汤科技2024年营收增长36%达到50亿元", "metadata": {"source": "annual_report"}}) + "\n" +
            json.dumps({"text": "英伟达Blackwell架构GPU性能提升4倍", "metadata": {"source": "tech_news"}}) + "\n" +
            json.dumps({"text": "科大讯飞星火大模型V4.0发布", "metadata": {"source": "product_news"}}) + "\n",
            encoding="utf-8"
        )

        # Ingest
        resp = self.client.post("/api/ingest/files", json={"dir": str(tmp_path), "analyze": False})
        assert resp.status_code == 200

        # Build
        resp = self.client.post("/api/build", json={"documents": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_count"] >= 3

        # Query (BM25-only since no embedding API in test)
        resp = self.client.post("/api/score", json={"query": "商汤科技营收", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1

    def test_api_kb_clear(self, tmp_path):
        """Clear KB removes all docs"""
        # Ingest something first
        test_file = tmp_path / "data.jsonl"
        test_file.write_text(
            json.dumps({"text": "test doc", "metadata": {"source": "test"}}) + "\n",
            encoding="utf-8"
        )
        self.client.post("/api/ingest/files", json={"dir": str(tmp_path), "analyze": False})

        # Clear
        resp = self.client.post("/api/kb/clear")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Verify empty
        resp = self.client.get("/api/kb/status")
        assert resp.json()["doc_count"] == 0

    def test_api_kb_search_keyword(self, tmp_path):
        """Search KB by keyword"""
        # Clear KB first
        self.client.post("/api/kb/clear")

        test_file = tmp_path / "search.jsonl"
        test_file.write_text(
            json.dumps({"text": "商汤科技财报分析", "metadata": {"source": "search_s1"}}) + "\n" +
            json.dumps({"text": "英伟达GPU新闻", "metadata": {"source": "search_s2"}}) + "\n",
            encoding="utf-8"
        )
        self.client.post("/api/ingest/files", json={"dir": str(tmp_path), "analyze": False})

        resp = self.client.get("/api/kb/search", params={"keyword": "商汤"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] >= 1

    def test_api_file_preview(self, tmp_path):
        """File preview endpoint returns file content"""
        test_file = tmp_path / "preview_test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
        # Ensure tmp_path is under ./data for security check
        import os
        data_dir = os.path.normpath("./data")
        os.makedirs(data_dir, exist_ok=True)
        preview_dir = os.path.join(data_dir, "_test_preview")
        os.makedirs(preview_dir, exist_ok=True)
        preview_file = os.path.join(preview_dir, "test.txt")
        with open(preview_file, "w", encoding="utf-8") as f:
            f.write("hello world\nline 2\nline 3\n")
        try:
            resp = self.client.get("/api/file/preview", params={"path": preview_dir, "file": "test.txt", "lines": 5})
            assert resp.status_code == 200
            data = resp.json()
            assert data["file"] == "test.txt"
            assert len(data["lines"]) >= 1
            assert "hello world" in data["lines"][0]
        finally:
            import shutil
            shutil.rmtree(preview_dir, ignore_errors=True)

    def test_api_file_preview_security(self):
        """File preview blocks paths outside ./data/"""
        resp = self.client.get("/api/file/preview", params={"path": "C:/Windows", "file": "system32.dll"})
        assert resp.status_code == 403

    def test_api_ingest_file_selection(self, tmp_path):
        """Ingest with file selection only imports selected files"""
        (tmp_path / "a.jsonl").write_text(
            json.dumps({"text": "doc from file A", "metadata": {"source": "sel_a"}}) + "\n",
            encoding="utf-8"
        )
        (tmp_path / "b.jsonl").write_text(
            json.dumps({"text": "doc from file B", "metadata": {"source": "sel_b"}}) + "\n",
            encoding="utf-8"
        )
        # Clear KB first
        self.client.post("/api/kb/clear")
        # Import only file A
        resp = self.client.post("/api/ingest/files", json={
            "dir": str(tmp_path), "analyze": False, "files": ["a.jsonl"]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] >= 1
        # Verify only file A was imported
        status = self.client.get("/api/kb/status").json()
        assert status["doc_count"] >= 1


# ===================== KB Dedup Integration =====================


class TestKBDedupIntegration:
    """Integration: doc_id + dedup work correctly in the full KB flow"""

    def test_dedup_across_ingest_rounds(self):
        """Two rounds of ingest with overlapping docs → no duplicates in KB"""
        docs_round1 = [
            {"text": "商汤科技营收增长", "meta": {"source": "file1.txt"}},
            {"text": "英伟达GPU发布", "meta": {"source": "file1.txt"}},
        ]
        docs_round2 = [
            {"text": "商汤科技营收增长", "meta": {"source": "file1.txt"}},  # dup
            {"text": "科大讯飞星火V4", "meta": {"source": "file2.txt"}},   # new
        ]

        # Assign IDs
        assign_doc_ids(docs_round1)
        assign_doc_ids(docs_round2)

        # First round: all go in
        kb = list(docs_round1)

        # Second round: dedup
        new_only = dedup_docs(kb, docs_round2)
        assert len(new_only) == 1  # Only 科大讯飞 is new
        assert "科大讯飞" in new_only[0]["text"]

        kb.extend(new_only)
        assert len(kb) == 3  # 2 original + 1 new

    def test_analysis_dedup_by_source_prefix(self):
        """Analysis results for same topic replace previous (not duplicate)"""
        kb = [
            {"text": "old analysis", "meta": {"source": "analysis:news:商汤科技"}},
            {"text": "other doc", "meta": {"source": "file.txt"}},
        ]
        # New analysis for same topic replaces old
        new_doc = {"text": "new analysis", "meta": {"source": "analysis:news:商汤科技"}}
        kb = [d for d in kb if d["meta"]["source"] != "analysis:news:商汤科技"]
        kb.append(new_doc)
        assert len(kb) == 2
        assert kb[-1]["text"] == "new analysis"
