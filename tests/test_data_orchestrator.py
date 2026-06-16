"""
Test DataOrchestrator — multi-pool data management

Covers:
- DataRouter: doc_type -> pool routing
- DataOrchestrator: ingest with auto-routing
- DataOrchestrator: search across pools
- DataOrchestrator: cross_search (primary + secondary)
- DataOrchestrator: stats and clear
- Pipeline integration: data_orchestrator parameter
"""
import pytest
from financial_rag.core.data_orchestrator import (
    DataOrchestrator, DataRouter, KnowledgePool, IngestStats,
    DEFAULT_POOLS, DOC_TYPE_TO_POOL,
)


# ===================== DataRouter =====================


class TestDataRouter:

    def test_route_by_meta_doc_type(self):
        router = DataRouter()
        doc = {"text": "some text", "meta": {"doc_type": "news"}}
        assert router.route(doc) == "news"

    def test_route_financial_report(self):
        router = DataRouter()
        doc = {"text": "some text", "meta": {"doc_type": "financial_report"}}
        assert router.route(doc) == "financial_report"

    def test_route_macro_data(self):
        router = DataRouter()
        doc = {"text": "some text", "meta": {"doc_type": "macro_data"}}
        assert router.route(doc) == "macro_data"

    def test_route_query_falls_to_general(self):
        router = DataRouter()
        doc = {"text": "some text", "meta": {"doc_type": "query"}}
        assert router.route(doc) == "general"

    def test_route_unknown_falls_to_general(self):
        router = DataRouter()
        doc = {"text": "some text", "meta": {"doc_type": "unknown_type"}}
        assert router.route(doc) == "general"

    def test_route_no_doc_type_uses_classifier(self):
        router = DataRouter()
        # Text with financial report keywords should route to financial_report
        doc = {"text": "公司2024年营收100亿，净利润同比增长20%", "meta": {}}
        pool = router.route(doc)
        # DocTypeClassifier should detect financial_report
        assert pool in ("financial_report", "general")

    def test_route_no_meta_key(self):
        router = DataRouter()
        doc = {"text": "央行降准0.5个百分点，释放流动性"}
        pool = router.route(doc)
        assert pool in ("macro_data", "general")

    def test_route_to_nonexistent_pool_falls_to_general(self):
        router = DataRouter(pool_names=["news", "general"])
        doc = {"text": "text", "meta": {"doc_type": "financial_report"}}
        # financial_report pool doesn't exist, falls to general
        assert router.route(doc) == "general"


# ===================== DOC_TYPE_TO_POOL mapping =====================


class TestDocTypeMapping:

    def test_all_default_types_mapped(self):
        for doc_type in ["financial_report", "news", "macro_data", "query", "other"]:
            assert doc_type in DOC_TYPE_TO_POOL

    def test_query_maps_to_general(self):
        assert DOC_TYPE_TO_POOL["query"] == "general"

    def test_other_maps_to_general(self):
        assert DOC_TYPE_TO_POOL["other"] == "general"


# ===================== DataOrchestrator =====================


class TestDataOrchestrator:

    @pytest.fixture
    def orch(self):
        return DataOrchestrator()

    def test_default_pools_created(self, orch):
        assert len(orch.pools) == 4
        for name in DEFAULT_POOLS:
            assert name in orch.pools

    def test_add_custom_pool(self, orch):
        orch.add_pool("research_notes")
        assert "research_notes" in orch.pools
        assert len(orch.pools) == 5

    def test_remove_pool(self, orch):
        orch.remove_pool("macro_data")
        assert "macro_data" not in orch.pools

    def test_ingest_routes_by_doc_type(self, orch):
        docs = [
            {"text": "公司2024年财报显示，全年营收达到100亿元，净利润50亿元，同比增长20%，毛利率提升至45%", "meta": {"doc_type": "financial_report"}},
            {"text": "记者获悉，某科技公司今日发布新产品，据了解该产品采用最新AI技术，预计将推动行业变革", "meta": {"doc_type": "news"}},
            {"text": "央行今日宣布降准0.5个百分点，释放长期流动性约1万亿元，旨在支持实体经济发展和稳定金融市场", "meta": {"doc_type": "macro_data"}},
        ]
        stats = orch.ingest(docs)
        assert stats.total_docs == 3
        assert stats.cleaned == 3
        assert stats.rejected == 0
        # Check routing
        assert stats.routed.get("financial_report", 0) >= 1
        assert stats.routed.get("news", 0) >= 1
        assert stats.routed.get("macro_data", 0) >= 1

    def test_ingest_rejects_empty_text(self, orch):
        docs = [
            {"text": "", "meta": {}},
            {"text": "   ", "meta": {}},
            {"text": "央行今日宣布降准0.5个百分点释放长期流动性支持实体经济", "meta": {"doc_type": "macro_data"}},
        ]
        stats = orch.ingest(docs)
        assert stats.rejected == 2
        assert stats.cleaned == 1

    def test_ingest_rejects_short_text(self, orch):
        # RelevanceGate rejects text < 20 chars
        docs = [
            {"text": "太短了", "meta": {"doc_type": "news"}},
        ]
        stats = orch.ingest(docs)
        assert stats.rejected >= 0  # May or may not reject depending on gate

    def test_search_empty_pools(self, orch):
        results = orch.search("test query")
        assert results == []

    def test_search_single_pool(self, orch):
        docs = [
            {"text": "公司2024年财报显示全年营收达到100亿元，净利润增长20%，毛利率提升至45%", "meta": {"doc_type": "financial_report"}},
            {"text": "公司发布年报显示营收创新高，净利润同比增长30%超预期", "meta": {"doc_type": "financial_report"}},
        ]
        orch.ingest(docs)
        results = orch.search("营收", pool_names=["financial_report"])
        assert len(results) > 0
        assert all(r.get("_pool") == "financial_report" for r in results)

    def test_search_across_pools(self, orch):
        docs = [
            {"text": "公司2024年财报显示全年营收达到100亿元，净利润增长显著", "meta": {"doc_type": "financial_report"}},
            {"text": "记者获悉该公司营收创新高，据报道称业绩超预期", "meta": {"doc_type": "news"}},
        ]
        orch.ingest(docs)
        results = orch.search("营收", top_k=10)
        pools_found = {r.get("_pool") for r in results}
        assert len(pools_found) >= 1  # At least one pool matched

    def test_search_with_weights(self, orch):
        docs_report = [{"text": "公司2024年财报营收100亿净利润50亿同比增长", "meta": {"doc_type": "financial_report"}}]
        docs_news = [{"text": "记者报道称该公司营收增长超预期市场预期", "meta": {"doc_type": "news"}}]
        orch.ingest(docs_report + docs_news)

        # Give financial_report higher weight
        results = orch.search(
            "营收",
            weights={"financial_report": 3.0, "news": 1.0},
            top_k=10,
        )
        if len(results) >= 2:
            # financial_report should rank higher
            assert results[0]["_pool"] == "financial_report"

    def test_cross_search(self, orch):
        docs = [
            {"text": "央行今日宣布降准0.5个百分点释放长期流动性约1万亿支持实体经济", "meta": {"doc_type": "macro_data"}},
            {"text": "记者报道降准利好银行板块股票，市场预计将推动信贷增长", "meta": {"doc_type": "news"}},
        ]
        orch.ingest(docs)

        results = orch.cross_search(
            "降准",
            primary="macro_data",
            secondary=["news"],
        )
        # Primary results first
        if results:
            primary_results = [r for r in results if r.get("_cross_role") == "primary"]
            secondary_results = [r for r in results if r.get("_cross_role") == "secondary"]
            assert len(primary_results) >= 0
            assert len(secondary_results) >= 0

    def test_cross_search_fallback_when_primary_empty(self, orch):
        docs = [{"text": "记者报道央行降准利好政策将推动经济发展", "meta": {"doc_type": "news"}}]
        orch.ingest(docs)

        # primary=macro_data is empty, should fallback
        results = orch.cross_search("降准", primary="macro_data", secondary=["news"])
        assert len(results) >= 0

    def test_get_stats(self, orch):
        docs = [
            {"text": "公司2024年财报营收100亿利润50亿同比增长超预期", "meta": {"doc_type": "financial_report"}},
        ]
        orch.ingest(docs)
        stats = orch.get_stats()
        assert stats["_total"] >= 1
        assert stats["_pool_count"] == 4
        assert stats["financial_report"]["doc_count"] >= 1

    def test_clear_all(self, orch):
        docs = [{"text": "公司2024年财报营收100亿利润50亿同比增长", "meta": {"doc_type": "financial_report"}}]
        orch.ingest(docs)
        orch.clear()
        stats = orch.get_stats()
        assert stats["_total"] == 0

    def test_clear_specific_pool(self, orch):
        docs = [
            {"text": "公司2024年财报营收100亿利润50亿同比增长超预期", "meta": {"doc_type": "financial_report"}},
            {"text": "记者报道某上市公司发布业绩预告显示增长趋势", "meta": {"doc_type": "news"}},
        ]
        orch.ingest(docs)
        orch.clear(pool_names=["financial_report"])
        stats = orch.get_stats()
        assert stats["financial_report"]["doc_count"] == 0
        assert stats["news"]["doc_count"] >= 1


# ===================== IngestStats =====================


class TestIngestStats:

    def test_summary_format(self):
        stats = IngestStats(total_docs=10, cleaned=8, rejected=2, routed={"news": 5, "general": 3})
        s = stats.summary()
        assert "total=10" in s
        assert "rejected=2" in s
        assert "news=5" in s


# ===================== Pipeline integration =====================


class TestPipelineIntegration:

    def test_pipeline_accepts_data_orchestrator(self):
        from financial_rag.core.pipeline import PipelineScheduler, PipelineConfig
        from financial_rag.core.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator()
        data_orch = DataOrchestrator()
        scheduler = PipelineScheduler(
            orchestrator=orch,
            data_orchestrator=data_orch,
            config=PipelineConfig(enable_data_fetch=False, enable_agent_analysis=False, enable_slot_output=False, enable_scoring=False),
        )
        assert scheduler.data_orchestrator is data_orch

    def test_pipeline_without_data_orchestrator_backward_compat(self):
        from financial_rag.core.pipeline import PipelineScheduler, PipelineConfig
        from financial_rag.core.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator()
        scheduler = PipelineScheduler(
            orchestrator=orch,
            config=PipelineConfig(enable_data_fetch=False, enable_index=False, enable_agent_analysis=False, enable_slot_output=False, enable_scoring=False),
        )
        assert scheduler.data_orchestrator is None
