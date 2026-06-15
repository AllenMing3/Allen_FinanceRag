"""
Test new tool modules — scoring_tools, coordinator_tools, report_tools, event_impact_tools

Covers:
- scoring_tools: evaluate_pipeline_quality, check_hallucination, generate_score_report
- coordinator_tools: classify_query_intent, select_agent_chain
- report_tools: synthesize_report (fallback mode, no LLM)
- event_impact_tools: fallback_assess
- STOCK_MAP shared data location
"""
import pytest


# ===================== Scoring Tools =====================


class TestEvaluatePipelineQuality:

    def test_all_phases(self):
        from financial_rag.tools.scoring_tools import evaluate_pipeline_quality
        result = evaluate_pipeline_quality(
            fetched_data=[{"x": 1}, {"x": 2}],
            retrieved_items=[{"text": "doc1"}],
            agent_results=[{"success": True, "agent_name": "A1"}],
            fill_stats={"filled_slots": 3, "total_slots": 5, "template_name": "default"},
        )
        assert result["total_stages"] == 4
        assert "stages" in result
        assert "summary" in result
        # Each stage should have score and elapsed_ms
        for s in result["stages"]:
            assert "score" in s
            assert "elapsed_ms" in s

    def test_empty_phases(self):
        from financial_rag.tools.scoring_tools import evaluate_pipeline_quality
        result = evaluate_pipeline_quality(
            fetched_data=[],
            retrieved_items=[],
        )
        assert result["total_stages"] == 2
        # Empty data → low scores
        for s in result["stages"]:
            assert s["score"] < 0.5

    def test_no_data(self):
        from financial_rag.tools.scoring_tools import evaluate_pipeline_quality
        result = evaluate_pipeline_quality()
        assert result["total_stages"] == 0


class TestCheckHallucination:

    def test_basic_check(self):
        from financial_rag.tools.scoring_tools import check_hallucination
        result = check_hallucination(
            output_text="商汤科技2024年营收50.3亿元",
            source_items=[{"text": "商汤科技实现营业收入50.3亿元"}],
        )
        assert "overall_score" in result
        assert "risk" in result

    def test_no_sources(self):
        from financial_rag.tools.scoring_tools import check_hallucination
        result = check_hallucination(output_text="某段分析文本")
        assert "risk" in result


class TestGenerateScoreReport:

    def test_full_report(self):
        from financial_rag.tools.scoring_tools import generate_score_report
        pipeline_scores = {
            "stages": [
                {"stage": "fetch", "score": 0.9, "elapsed_ms": 100},
                {"stage": "index", "score": 0.85, "elapsed_ms": 200},
            ],
            "total_stages": 2,
        }
        hallucination_check = {
            "risk": "low",
            "overall_score": 0.95,
            "checks": {"numeric_citation": {"score": 0.9}},
            "warnings": [],
        }
        result = generate_score_report(
            pipeline_scores=pipeline_scores,
            hallucination_check=hallucination_check,
            query="测试查询",
        )
        report = result["report"]
        assert "全链路评分报告" in report
        assert "fetch" in report
        assert "测试查询" in report

    def test_empty_scores(self):
        from financial_rag.tools.scoring_tools import generate_score_report
        result = generate_score_report(query="test")
        assert "report" in result


# ===================== Coordinator Tools =====================


class TestClassifyQueryIntent:

    def test_kline_intent(self):
        from financial_rag.tools.coordinator_tools import classify_query_intent
        result = classify_query_intent("茅台走势分析")
        assert result["intent"] == "kline"
        assert result["confidence"] > 0

    def test_event_intent(self):
        from financial_rag.tools.coordinator_tools import classify_query_intent
        result = classify_query_intent("2024年6月1日发生了什么大事")
        assert result["intent"] == "event_impact"

    def test_general_intent(self):
        from financial_rag.tools.coordinator_tools import classify_query_intent
        result = classify_query_intent("随便聊聊")
        assert result["intent"] == "general"


class TestSelectAgentChain:

    def test_kline_chain(self):
        from financial_rag.tools.coordinator_tools import select_agent_chain
        result = select_agent_chain(intent="kline")
        assert "KLineAgent" in result["agent_chain"]
        assert "ScoringAgent" in result["agent_chain"]

    def test_event_chain(self):
        from financial_rag.tools.coordinator_tools import select_agent_chain
        result = select_agent_chain(intent="event_impact")
        assert "EventImpactAgent" in result["agent_chain"]

    def test_low_confidence_adds_ingestion(self):
        from financial_rag.tools.coordinator_tools import select_agent_chain
        result = select_agent_chain(intent="kline", confidence=0.2)
        assert "IngestionAgent" in result["agent_chain"]

    def test_unknown_intent_uses_default(self):
        from financial_rag.tools.coordinator_tools import select_agent_chain
        result = select_agent_chain(intent="unknown_type")
        assert len(result["agent_chain"]) > 0
        assert "ReportAgent" in result["agent_chain"]


# ===================== Report Tools =====================


class TestSynthesizeReport:

    def test_fallback_no_llm(self):
        from financial_rag.tools.report_tools import synthesize_report
        result = synthesize_report(
            query="测试查询",
            sources=[
                {"id": 1, "text": "商汤科技营收50亿", "title": "商汤年报", "source": "test"},
                {"id": 2, "text": "AI行业增长30%", "title": "行业报告", "source": "test"},
            ],
            metrics={"revenue": 50},
            entities=[{"type": "company", "data": {"name": "商汤科技"}}],
        )
        assert result["method"] == "fallback"
        report = result["report"]
        assert "title" in report
        assert "key_findings" in report

    def test_no_data(self):
        from financial_rag.tools.report_tools import synthesize_report
        result = synthesize_report(query="test")
        assert "error" in result


# ===================== Event Impact Tools =====================


class TestFallbackAssess:

    def test_bullish_keywords(self):
        from financial_rag.tools.event_impact_tools import _fallback_assess
        events = [
            {"title": "公司获得重大合同，业绩预增"},
            {"title": "获得政府补贴，利润大幅提升"},
        ]
        result = _fallback_assess(events)
        assert "assessments" in result
        assert result["overall_label"] in ["综合利好", "综合利空", "综合中性"]

    def test_bearish_keywords(self):
        from financial_rag.tools.event_impact_tools import _fallback_assess
        events = [
            {"title": "公司亏损严重，面临退市风险"},
            {"title": "高管辞职，股价暴跌"},
        ]
        result = _fallback_assess(events)
        assert "assessments" in result

    def test_empty_events(self):
        from financial_rag.tools.event_impact_tools import _fallback_assess
        result = _fallback_assess([])
        assert result["overall_label"] == "综合中性"


# ===================== STOCK_MAP Shared Data =====================


class TestStockMap:

    def test_stock_map_in_kline_tools(self):
        from financial_rag.tools.kline_tools import STOCK_MAP
        assert "茅台" in STOCK_MAP
        assert STOCK_MAP["茅台"] == ("600519.SH", "贵州茅台")

    def test_stock_map_reexported_from_agent(self):
        """Backward compat: kline_agent re-exports STOCK_MAP"""
        from financial_rag.agents.kline_agent import STOCK_MAP
        assert "茅台" in STOCK_MAP

    def test_stock_map_in_tools_init(self):
        """tools/__init__.py exports STOCK_MAP"""
        from financial_rag.tools import STOCK_MAP
        assert "比亚迪" in STOCK_MAP
