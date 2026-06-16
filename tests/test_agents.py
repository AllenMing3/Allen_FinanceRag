"""
Test agents — IngestionAgent + AnalysisAgent with tool registry (regex fallback)

Tests the agent chain: IngestionAgent → AnalysisAgent (extraction path).
No LLM needed — exercises the function calling plumbing.
"""
import pytest
from financial_rag.core.base import AgentContext
from financial_rag.agents.ingestion_agent import IngestionAgent
from financial_rag.agents.analysis_agent import AnalysisAgent


class TestIngestionAgent:
    """IngestionAgent — text ingestion with tool calling"""

    def _make_agent(self, registry, executor):
        agent = IngestionAgent()
        agent.bind_tools(registry, executor)
        return agent

    def test_text_ingestion(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input=ai_financial_text)
        result = agent.run(ctx)

        assert result.success
        assert result.agent_name == "IngestionAgent"
        assert result.data is not None
        assert len(result.data) >= 1
        assert "text" in result.data[0]
        assert "meta" in result.data[0]

    def test_metadata_extraction(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input=ai_financial_text)
        result = agent.run(ctx)

        meta = result.context_updates.get("metadata", {})
        assert meta["doc_count"] >= 1
        assert "metadata_score" in meta

    def test_news_text(self, registry, executor, ai_news_text):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input=ai_news_text)
        result = agent.run(ctx)
        assert result.success

    def test_funding_text(self, registry, executor, ai_funding_text):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input=ai_funding_text)
        result = agent.run(ctx)
        assert result.success

    def test_empty_input(self, registry, executor):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input="")
        result = agent.run(ctx)
        assert result.agent_name == "IngestionAgent"

    def test_short_text(self, registry, executor):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input="商汤科技今日发布新模型")
        result = agent.run(ctx)
        assert result.success

    def test_evaluate_metadata_api(self, registry, executor):
        agent = self._make_agent(registry, executor)
        docs = [{"text": "test", "meta": {"source": "test", "date": "2024-01-01"}}]
        eval_result = agent.evaluate_metadata(docs)
        assert "score" in eval_result
        assert "fields_found" in eval_result
        assert eval_result["fields_expected"] == len(agent.EXPECTED_METADATA_FIELDS)


class TestAnalysisAgentExtraction:
    """AnalysisAgent — extraction path (metrics + entities + queries)"""

    def _make_agent(self, registry, executor):
        agent = AnalysisAgent()
        agent.bind_tools(registry, executor)
        return agent

    def test_extract_from_financial_report(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_financial_text}]
        ctx = AgentContext(
            parsed_data=docs,
            raw_input="分析财报",
            metadata={"intent": "report"},
        )
        result = agent.run(ctx)

        assert result.success
        assert result.agent_name == "AnalysisAgent"
        data = result.data
        assert "report" in data or "_scores" in data

    def test_extract_scores(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_financial_text}]
        ctx = AgentContext(
            parsed_data=docs, raw_input="分析",
            metadata={"intent": "report"},
        )
        result = agent.run(ctx)
        scores = result.data.get("_scores", {})
        assert "extraction" in scores
        assert "query_rewrite" in scores

    def test_no_documents(self, registry, executor):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(
            parsed_data=[], raw_input="分析",
            metadata={"intent": "report"},
        )
        result = agent.run(ctx)
        # No docs → report still generates with empty sources
        assert result.agent_name == "AnalysisAgent"


class TestAgentChain:
    """Full agent chain: IngestionAgent → AnalysisAgent"""

    def test_ingest_then_analyze(self, registry, executor, ai_financial_text):
        """Simulate the real pipeline: ingest → pass context → analyze"""
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=ai_financial_text)
        ingest_result = ingest.run(ctx)
        assert ingest_result.success

        analyze = AnalysisAgent()
        analyze.bind_tools(registry, executor)
        analyze_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", []),
            raw_input=ai_financial_text,
            metadata={"intent": "report"},
        )
        analyze_result = analyze.run(analyze_ctx)
        assert analyze_result.success

    def test_chain_with_news(self, registry, executor, ai_news_text):
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=ai_news_text)
        ingest_result = ingest.run(ctx)
        assert ingest_result.success

        analyze = AnalysisAgent()
        analyze.bind_tools(registry, executor)
        analyze_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", []),
            raw_input=ai_news_text,
            metadata={"intent": "news"},
        )
        analyze_result = analyze.run(analyze_ctx)
        assert analyze_result.success


class TestAgentChainWithLongArticles:
    """Full agent chain with realistic long-form AI-sector articles"""

    def test_long_article_chain(self, registry, executor, long_article_sensetime):
        """~2000字商汤年报 → IngestionAgent → AnalysisAgent"""
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=long_article_sensetime)
        ingest_result = ingest.run(ctx)
        assert ingest_result.success

        analyze = AnalysisAgent()
        analyze.bind_tools(registry, executor)
        analyze_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", []),
            raw_input=long_article_sensetime,
            metadata={"intent": "report"},
        )
        analyze_result = analyze.run(analyze_ctx)
        assert analyze_result.success

    def test_all_long_articles_chain(self, registry, executor, all_long_articles):
        """All long articles through full chain"""
        for article in all_long_articles:
            assert len(article) > 500

            ingest = IngestionAgent()
            ingest.bind_tools(registry, executor)
            ctx = AgentContext(raw_input=article)
            r = ingest.run(ctx)
            assert r.success

            analyze = AnalysisAgent()
            analyze.bind_tools(registry, executor)
            analyze_ctx = AgentContext(
                parsed_data=r.context_updates.get("parsed_data", []),
                raw_input=article,
                metadata={"intent": "report"},
            )
            result = analyze.run(analyze_ctx)
            assert result.success
