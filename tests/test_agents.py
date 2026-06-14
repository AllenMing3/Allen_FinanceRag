"""
Test agents — IngestionAgent + ExtractionAgent with tool registry (regex fallback)

Tests the full agent chain: agent.process() → call_tool() → regex fallback.
No LLM needed — exercises the function calling plumbing.
"""
import pytest
from financial_rag.core.base import AgentContext
from financial_rag.agents.ingestion_agent import IngestionAgent
from financial_rag.agents.extraction_agent import ExtractionAgent


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
        assert "metadata" in result.data[0]

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
        # Empty input → no documents → still runs but success=False
        assert result.agent_name == "IngestionAgent"

    def test_short_text(self, registry, executor):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(raw_input="商汤科技今日发布新模型")
        result = agent.run(ctx)
        assert result.success

    def test_evaluate_metadata_api(self, registry, executor):
        agent = self._make_agent(registry, executor)
        docs = [{"text": "test", "metadata": {"source": "test", "date": "2024-01-01"}}]
        eval_result = agent.evaluate_metadata(docs)
        assert "score" in eval_result
        assert "fields_found" in eval_result
        assert eval_result["fields_expected"] == len(agent.EXPECTED_METADATA_FIELDS)


class TestExtractionAgent:
    """ExtractionAgent — metric + entity extraction with tool calling"""

    def _make_agent(self, registry, executor):
        agent = ExtractionAgent()
        agent.bind_tools(registry, executor)
        return agent

    def test_extract_from_financial_report(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_financial_text}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)

        assert result.success
        assert result.agent_name == "ExtractionAgent"
        data = result.data
        assert "metrics" in data
        assert "entities" in data
        assert "queries" in data

    def test_extract_scores(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_financial_text}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)

        scores = result.data.get("_scores", {})
        assert "extraction" in scores
        assert "query_rewrite" in scores
        assert 0 <= scores["extraction"] <= 1
        assert 0 <= scores["query_rewrite"] <= 1

    def test_extract_confidence(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_financial_text}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)

        confidence = result.data.get("_confidence", {})
        assert "metrics" in confidence
        assert "entities" in confidence

    def test_extract_from_news(self, registry, executor, ai_news_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_news_text}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)
        assert result.success

    def test_extract_from_funding(self, registry, executor, ai_funding_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_funding_text}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)
        assert result.success

    def test_no_documents(self, registry, executor):
        agent = self._make_agent(registry, executor)
        ctx = AgentContext(parsed_data=[])
        result = agent.run(ctx)
        assert not result.success
        assert "无文档" in result.message

    def test_empty_text(self, registry, executor):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ""}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)
        assert not result.success

    def test_context_updates(self, registry, executor, ai_financial_text):
        agent = self._make_agent(registry, executor)
        docs = [{"text": ai_financial_text}]
        ctx = AgentContext(parsed_data=docs)
        result = agent.run(ctx)

        updates = result.context_updates
        assert "extracted_features" in updates
        assert "intermediate_findings" in updates
        assert len(updates["intermediate_findings"]) >= 1


class TestAgentChain:
    """Full agent chain: IngestionAgent → ExtractionAgent"""

    def test_ingest_then_extract(self, registry, executor, ai_financial_text):
        """Simulate the real pipeline: ingest → pass context → extract"""
        # Step 1: Ingestion
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=ai_financial_text)
        ingest_result = ingest.run(ctx)
        assert ingest_result.success

        # Step 2: Feed ingestion output to extraction
        extract = ExtractionAgent()
        extract.bind_tools(registry, executor)
        extract_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", [])
        )
        extract_result = extract.run(extract_ctx)
        assert extract_result.success
        assert "metrics" in extract_result.data
        assert "entities" in extract_result.data
        assert "queries" in extract_result.data

    def test_chain_with_news(self, registry, executor, ai_news_text):
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=ai_news_text)
        ingest_result = ingest.run(ctx)
        assert ingest_result.success

        extract = ExtractionAgent()
        extract.bind_tools(registry, executor)
        extract_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", [])
        )
        extract_result = extract.run(extract_ctx)
        assert extract_result.success

    def test_chain_with_multiple_docs(self, registry, executor, ai_financial_text, ai_news_text):
        """Multi-doc ingestion → single extraction"""
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)

        # Ingest first doc
        ctx1 = AgentContext(raw_input=ai_financial_text)
        r1 = ingest.run(ctx1)
        docs1 = r1.context_updates.get("parsed_data", [])

        # Ingest second doc
        ctx2 = AgentContext(raw_input=ai_news_text)
        r2 = ingest.run(ctx2)
        docs2 = r2.context_updates.get("parsed_data", [])

        # Combine and extract
        extract = ExtractionAgent()
        extract.bind_tools(registry, executor)
        all_docs = docs1 + docs2
        extract_ctx = AgentContext(parsed_data=all_docs)
        extract_result = extract.run(extract_ctx)
        assert extract_result.success


class TestAgentChainWithLongArticles:
    """Full agent chain with realistic long-form AI-sector articles"""

    def test_long_article_ingest_extract(self, registry, executor, long_article_sensetime):
        """~2000字商汤年报 → IngestionAgent → ExtractionAgent"""
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=long_article_sensetime)
        ingest_result = ingest.run(ctx)
        assert ingest_result.success

        extract = ExtractionAgent()
        extract.bind_tools(registry, executor)
        extract_ctx = AgentContext(
            parsed_data=ingest_result.context_updates.get("parsed_data", [])
        )
        extract_result = extract.run(extract_ctx)
        assert extract_result.success
        # Should extract metrics from long article
        metrics = extract_result.data.get("metrics", {})
        metric_keys = [k for k in metrics if not k.startswith("_")]
        assert len(metric_keys) >= 1

    def test_long_article_nvidia(self, registry, executor, long_article_nvidia):
        """~2000字英伟达架构解析 → full chain"""
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=long_article_nvidia)
        r = ingest.run(ctx)
        assert r.success

        extract = ExtractionAgent()
        extract.bind_tools(registry, executor)
        extract_ctx = AgentContext(
            parsed_data=r.context_updates.get("parsed_data", [])
        )
        result = extract.run(extract_ctx)
        assert result.success

    def test_long_article_funding(self, registry, executor, long_article_funding):
        """~1500字融资盘点 → full chain"""
        ingest = IngestionAgent()
        ingest.bind_tools(registry, executor)
        ctx = AgentContext(raw_input=long_article_funding)
        r = ingest.run(ctx)
        assert r.success

        extract = ExtractionAgent()
        extract.bind_tools(registry, executor)
        extract_ctx = AgentContext(
            parsed_data=r.context_updates.get("parsed_data", [])
        )
        result = extract.run(extract_ctx)
        assert result.success
        # Should find entities (companies, amounts)
        entities = result.data.get("entities", {})
        assert isinstance(entities, dict)

    def test_all_long_articles_chain(self, registry, executor, all_long_articles):
        """All 3 long articles through full chain"""
        for article in all_long_articles:
            assert len(article) > 500

            ingest = IngestionAgent()
            ingest.bind_tools(registry, executor)
            ctx = AgentContext(raw_input=article)
            r = ingest.run(ctx)
            assert r.success

            extract = ExtractionAgent()
            extract.bind_tools(registry, executor)
            extract_ctx = AgentContext(
                parsed_data=r.context_updates.get("parsed_data", [])
            )
            result = extract.run(extract_ctx)
            assert result.success
            assert result.data["metrics"] is not None
            assert result.data["entities"] is not None
