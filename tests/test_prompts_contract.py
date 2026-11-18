"""
Prompt 契约回归锁 — prompts.py 质量加固

离线验证（无 LLM / 无网络 / 无 Phoenix）：
- 占位符契约：活模板用调用方实际参数 .format() 不抛 KeyError，且注入值真实出现在结果里
- schema 字段锁：输出 JSON 字段名不得漂移（调用方解析依赖）
- 引用编号契约：NEWS_SYNTHESIS 沿用来源 id 不得重编号；FOLLOWUP [N] 从 1 开始
- 死代码锁：FINANCIAL_NEWS_EXTRACTION_* 已删除，防复活
- FOLLOWUP 两件套：<edge_cases> / <negative_examples>

运行: python -m pytest tests/test_prompts_contract.py -v
"""
import pytest

from financial_rag import prompts

pytestmark = pytest.mark.unit


# ===================== 占位符契约（调用方实际参数） =====================

class TestPlaceholderContract:
    def test_metadata_prompt_formats(self):
        out = prompts.METADATA_EXTRACTION_PROMPT.format(text="样例文本注入")
        assert "样例文本注入" in out

    def test_metrics_prompt_formats(self):
        out = prompts.FINANCIAL_METRICS_EXTRACTION_PROMPT.format(text="样例文本注入")
        assert "样例文本注入" in out

    def test_entity_prompt_formats(self):
        out = prompts.ENTITY_EXTRACTION_PROMPT.format(text="样例文本注入")
        assert "样例文本注入" in out

    def test_synthesis_prompt_formats(self):
        out = prompts.NEWS_SYNTHESIS_PROMPT.format(
            query="算力查询注入", sources="[1] 来源注入", metrics="{}", entities="{}",
        )
        assert "算力查询注入" in out
        assert "[1] 来源注入" in out

    def test_followup_system_formats(self):
        out = prompts.FOLLOWUP_ANSWER_SYSTEM.format(
            assessment="利好注入", confidence="high",
            structured="{}", grounding="[1] 证据注入",
        )
        assert "利好注入" in out
        assert "[1] 证据注入" in out

    def test_image_prompt_no_placeholders(self):
        # 无占位符 → format 必须是恒等变换（新增活占位符即红）
        assert prompts.IMAGE_UNDERSTANDING_PROMPT.format() == prompts.IMAGE_UNDERSTANDING_PROMPT


# ===================== schema 字段锁（调用方解析依赖） =====================

@pytest.mark.critical
class TestSchemaFieldLock:
    def test_synthesis_schema_fields(self):
        tpl = prompts.NEWS_SYNTHESIS_SYSTEM
        for field in ("key_findings", "source_refs", "trend_analysis",
                      "sentiment", "contradictions", "summary"):
            assert field in tpl, f"NEWS_SYNTHESIS 缺字段 {field}"

    def test_metrics_schema_fields(self):
        tpl = prompts.FINANCIAL_METRICS_EXTRACTION_PROMPT
        for field in ("value", "unit", "yoy_growth", "period"):
            assert field in tpl, f"METRICS 缺字段 {field}"

    def test_entity_schema_fields(self):
        tpl = prompts.ENTITY_EXTRACTION_PROMPT
        for field in ("companies", "persons", "ai_models",
                      "chips_hardware", "event"):
            assert field in tpl, f"ENTITY 缺字段 {field}"

    def test_metadata_schema_fields(self):
        tpl = prompts.METADATA_EXTRACTION_PROMPT
        for field in ("source", "company", "date",
                      "fiscal_period", "currency", "doc_type"):
            assert field in tpl, f"METADATA 缺字段 {field}"


# ===================== 引用编号契约 =====================

@pytest.mark.critical
class TestCitationContract:
    def test_synthesis_uses_source_ids(self):
        # report_tools._format_sources 用 [{id}] 标注来源，
        # prompt 必须要求沿用该编号，不得自行重编号
        assert "不得自行重新编号" in prompts.NEWS_SYNTHESIS_SYSTEM

    def test_followup_citations_one_based(self):
        tpl = prompts.FOLLOWUP_ANSWER_SYSTEM
        assert "[N]" in tpl
        assert "从 1 开始" in tpl


# ===================== 死代码锁（防复活） =====================

class TestDeadCodeRemoved:
    def test_financial_news_extraction_removed(self):
        assert not hasattr(prompts, "FINANCIAL_NEWS_EXTRACTION_SYSTEM")
        assert not hasattr(prompts, "FINANCIAL_NEWS_EXTRACTION_PROMPT")


# ===================== FOLLOWUP 两件套补齐 =====================

class TestFollowupSections:
    def test_has_edge_cases(self):
        assert "<edge_cases>" in prompts.FOLLOWUP_ANSWER_SYSTEM

    def test_has_negative_examples(self):
        assert "<negative_examples>" in prompts.FOLLOWUP_ANSWER_SYSTEM


# ===================== 深度研判 prompt 契约（analysis.py 迁移） =====================

class TestAssessmentPlaceholderContract:
    def test_news_user_formats(self):
        kwargs = dict(text="新闻注入样例", doc_type="新闻报道", metrics="{}",
                      entities="{}", kb_grounding="(无)")
        for instruction in ("请输出 JSON 分析结果。", "请给出你的分析。"):
            out = prompts.NEWS_ASSESSMENT_USER.format(instruction=instruction, **kwargs)
            assert "新闻注入样例" in out
            assert instruction in out

    def test_topic_user_formats(self):
        kwargs = dict(topic="算力话题注入", news_count=3, news_preview="- 标题",
                      combined_text="详情注入样例", kb_grounding="(无)")
        for instruction in ("请输出 JSON 研判结果。", "请给出综合研判。"):
            out = prompts.TOPIC_ASSESSMENT_USER.format(instruction=instruction, **kwargs)
            assert "算力话题注入" in out
            assert "详情注入样例" in out
            assert instruction in out


@pytest.mark.critical
class TestAssessmentSchemaLock:
    def test_news_schema_fields(self):
        tpl = prompts.NEWS_ASSESSMENT_SYSTEM
        for field in ("verdict", "confidence", "impact", "key_signals",
                      "analysis", "risks", "watch_next"):
            assert field in tpl, f"NEWS_ASSESSMENT 缺字段 {field}"

    def test_topic_schema_fields(self):
        tpl = prompts.TOPIC_ASSESSMENT_SYSTEM
        for field in ("verdict", "confidence", "sub_topics", "key_players",
                      "sentiment_trend", "contrarian_signals",
                      "investment_implication", "risks"):
            assert field in tpl, f"TOPIC_ASSESSMENT 缺字段 {field}"


@pytest.mark.critical
class TestAssessmentExplainability:
    """可解释性契约：判定依据必须显式输出，证据必须逐字摘录（前端展示依赖）"""

    def test_news_reasoning_fields(self):
        tpl = prompts.NEWS_ASSESSMENT_SYSTEM
        for field in ("verdict_reasoning", "confidence_reasoning", "evidence"):
            assert field in tpl, f"NEWS_ASSESSMENT 缺可解释性字段 {field}"

    def test_topic_reasoning_fields(self):
        tpl = prompts.TOPIC_ASSESSMENT_SYSTEM
        for field in ("verdict_reasoning", "confidence_reasoning", "evidence"):
            assert field in tpl, f"TOPIC_ASSESSMENT 缺可解释性字段 {field}"

    def test_news_evidence_verbatim_discipline(self):
        tpl = prompts.NEWS_ASSESSMENT_SYSTEM
        assert "逐字摘录" in tpl
        assert "禁止编造" in tpl

    def test_topic_evidence_verbatim_discipline(self):
        tpl = prompts.TOPIC_ASSESSMENT_SYSTEM
        assert "逐字摘录" in tpl
        assert "禁止编造" in tpl


@pytest.mark.critical
class TestAssessmentFallbackMarkers:
    """降级路径【】标记是 _text_to_structured_* 解析器契约，逐字不动"""

    def test_news_fallback_markers(self):
        tpl = prompts.NEWS_ASSESSMENT_FALLBACK_SYSTEM
        for marker in ("【判断】", "【置信度】", "【关键信号】", "【影响分析】", "【风险提示】"):
            assert marker in tpl, f"news 降级缺标记 {marker}"

    def test_topic_fallback_markers(self):
        tpl = prompts.TOPIC_ASSESSMENT_FALLBACK_SYSTEM
        for marker in ("【行业动态】", "【综合分析】", "【投资启示】", "【风险提示】"):
            assert marker in tpl, f"topic 降级缺标记 {marker}"


@pytest.mark.critical
class TestAssessmentKBRules:
    def test_news_kb_text_reference_rule(self):
        tpl = prompts.NEWS_ASSESSMENT_SYSTEM
        assert "知识库背景" in tpl
        assert "不要标 [N]" in tpl

    def test_topic_kb_text_reference_rule(self):
        tpl = prompts.TOPIC_ASSESSMENT_SYSTEM
        assert "知识库背景" in tpl
        assert "不要标 [N]" in tpl


class TestInstructionPlaceholder:
    def test_news_user_has_instruction(self):
        assert "{instruction}" in prompts.NEWS_ASSESSMENT_USER

    def test_topic_user_has_instruction(self):
        assert "{instruction}" in prompts.TOPIC_ASSESSMENT_USER


@pytest.mark.critical
class TestSlotFillerPrompt:
    def test_slot_filler_system_honesty_rule(self):
        # 诚实拒答合规语义必须在场（评分器不罚诚实的前提）
        assert hasattr(prompts, "SLOT_FILLER_SYSTEM")
        assert "信息不足" in prompts.SLOT_FILLER_SYSTEM

    def test_slot_filler_system_anti_fabrication(self):
        assert "编造" in prompts.SLOT_FILLER_SYSTEM