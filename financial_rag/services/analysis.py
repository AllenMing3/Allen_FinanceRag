"""
Financial RAG — Analysis Service

Pure business logic for news analysis and topic research.
No HTTP, no FastAPI — just functions that take dependencies and return dicts.

Functions:
    analyze_news_text  — paste news → extract + KB + LLM verdict
    analyze_topic_research — search topic → fetch news + KB + LLM verdict
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# News text analysis (user pastes text → structured extraction + verdict)
# ---------------------------------------------------------------------------

def analyze_news_text(
    text: str,
    *,
    query: str = "",
    llm=None,
    retriever=None,
    kb_built: bool = False,
) -> dict:
    """Analyze pasted news text: structured extraction + KB context + bullish/bearish verdict.

    Args:
        text: raw news text
        query: optional search query for KB (defaults to text[:200])
        llm: DashScope LLM instance (None → rule-based fallback)
        retriever: HybridRetriever instance (None → skip KB search)
        kb_built: whether KB index is built

    Returns:
        dict with assessment, analysis, metrics, entities, doc_type, kb_sources
    """
    from financial_rag.tools.extraction_tools import (
        extract_financial_metrics, extract_entities, detect_document_type,
    )

    logger.info(f"[analyze_news] 开始: text={len(text)}字, query={query!r}, kb_built={kb_built}, llm={'yes' if llm else 'no'}")

    # 1. Structured extraction
    doc_type = detect_document_type(text)
    metrics = extract_financial_metrics(text=text)
    entities = extract_entities(text=text)
    logger.info(f"[analyze_news] 抽取完成: doc_type={doc_type}, "
                f"metrics_keys={list(metrics.keys())}, entities_keys={list(entities.keys())}")

    metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    entities_clean = {k: v for k, v in entities.items() if not k.startswith("_")}

    # 2. KB context retrieval
    kb_sources = _search_kb(retriever, query or text[:200], kb_built)
    logger.info(f"[analyze_news] KB 检索: {len(kb_sources)} sources")

    # 3. LLM assessment or rule-based fallback
    if llm:
        assessment, analysis = _llm_news_assessment(llm, text, doc_type, metrics_clean, entities_clean, kb_sources)
    else:
        assessment, analysis = _heuristic_assessment(doc_type, metrics_clean, entities_clean)

    logger.info(f"[analyze_news] 完成: assessment={assessment}, analysis_type={type(analysis).__name__}, analysis_len={len(str(analysis))}")
    return {
        "assessment": assessment,
        "analysis": analysis,
        "metrics": metrics_clean,
        "entities": entities_clean,
        "doc_type": doc_type,
        "kb_sources": kb_sources,
    }


# ---------------------------------------------------------------------------
# Topic research (user inputs topic → fetch news + KB + comprehensive verdict)
# ---------------------------------------------------------------------------

def analyze_topic_research(
    topic: str,
    *,
    max_news: int = 20,
    llm=None,
    retriever=None,
    kb_built: bool = False,
) -> dict:
    """Topic research: fetch news + query KB + LLM comprehensive assessment.

    Args:
        topic: research topic (e.g. "英伟达 Blackwell")
        max_news: max news items to fetch
        llm: DashScope LLM instance (None → basic summary)
        retriever: HybridRetriever instance
        kb_built: whether KB index is built

    Returns:
        dict with assessment, analysis, topic, news_count, news, kb_sources
    """
    # 1. Fetch news (mock-aware)
    logger.info(f"[analyze_topic] 开始: topic={topic!r}, max_news={max_news}, kb_built={kb_built}, llm={'yes' if llm else 'no'}")
    from financial_rag.config import is_mock_enabled
    if is_mock_enabled():
        from financial_rag.mock_data import mock_search_news
        mock_result = mock_search_news(topic, max_news=max_news)
        items = mock_result.get("items", [])
        logger.info(f"[Mock] Topic '{topic}': {len(items)} mock news items")
    else:
        from financial_rag.tools.news_tools import run_news_pipeline
        data = run_news_pipeline(llm=llm, query=topic, summarize=False, max_news=max_news)
        items = data.get("items", [])

    # 2. Combine news content
    combined_text = "\n\n".join(
        f"{item.get('title', '')}: {item.get('content', '')}"
        for item in items if item.get("title") or item.get("content")
    )[:6000]

    # 3. KB search
    search_query = topic
    if llm:
        try:
            from financial_rag.tools.news_tools import _extract_keywords
            keywords = _extract_keywords(llm, topic)
            search_query = keywords[0] if keywords else topic
        except Exception:
            pass
    kb_sources = _search_kb(retriever, search_query, kb_built)
    logger.info(f"[analyze_topic] 新闻 {len(items)} 条, KB {len(kb_sources)} sources")

    # 4. LLM assessment or fallback
    if llm:
        assessment, analysis = _llm_topic_assessment(llm, topic, items, combined_text, kb_sources)
    else:
        assessment = "neutral"
        analysis = f"话题: {topic}\n获取新闻: {len(items)} 条\n\n配置 DASHSCOPE_API_KEY 可获得 LLM 智能研判。"

    logger.info(f"[analyze_topic] 完成: assessment={assessment}, news_count={len(items)}")
    return {
        "assessment": assessment,
        "analysis": analysis,
        "topic": topic,
        "news_count": len(items),
        "news": [
            {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "publish_time": item.get("publish_time", ""),
                "content": item.get("content", "")[:200],
            }
            for item in items[:10]
        ],
        "kb_sources": kb_sources,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _search_kb(retriever, query: str, kb_built: bool, min_score: float = 0.4) -> list:
    """Search KB for relevant context documents.

    Only returns results above min_score threshold to avoid
    feeding irrelevant content (score < 0.4) to the LLM.
    """
    if not kb_built or not retriever:
        return []
    try:
        results, _ = retriever.search_with_scores(query, top_k=5)
        filtered = [
            {
                "text": it.get("text", ""),
                "source": it.get("meta", {}).get("source", ""),
                "score": round(it.get("score", 0), 4),
            }
            for it in results[:5]
            if it.get("score", 0) >= min_score
        ]
        if len(results) > 0:
            logger.info(f"[KB search] {len(results)} results, {len(filtered)} above threshold {min_score}")
        return filtered
    except Exception as e:
        logger.warning(f"KB search failed: {e}")
        return []


def _parse_verdict(text: str) -> str:
    """Extract bullish/bearish/neutral from LLM response text."""
    if "利好" in text:
        return "bullish"
    elif "利空" in text:
        return "bearish"
    return "neutral"


def _llm_news_assessment(llm, text: str, doc_type: str, metrics: dict, entities: dict, kb_sources: list) -> tuple:
    """LLM-powered news assessment with structured output — via LLMCaller."""
    from financial_rag.llm.caller import LLMCaller

    metrics_str = "\n".join(f"  - {k}: {v}" for k, v in metrics.items()) if metrics else "  (无)"
    entities_str = "\n".join(f"  - {k}: {v}" for k, v in entities.items()) if entities else "  (无)"
    kb_str = "\n".join(f"  - [{s['source']}] (相关度{s['score']:.2f}) {s['text'][:150]}" for s in kb_sources[:3]) if kb_sources else "  (无相关知识库背景)"

    system = """你是专业的 AI/科技行业分析师。请对以下新闻进行深度分析。

你必须输出以下格式：
【判断】利好 / 利空 / 中性（三选一）
【置信度】高 / 中 / 低
【关键信号】列出 2-4 个支撑你判断的关键事实
【影响分析】150-300字，说明为什么是利好/利空/中性，涉及哪些公司/行业
【风险提示】1-2 个需要关注的不确定因素

要求：
- 有依据、不空泛、站在投资者角度
- 基于提供的信息回答，不要编造任何数据或数字
- 如果知识库背景与新闻无关（相关度低于0.5），请忽略它，不要强行关联
- 如果信息不足，明确说明“数据不足，无法判断”
"""

    user = f"""新闻内容：
{text[:3000]}

文档类型：{doc_type}

抽取的结构化指标：
{metrics_str}

抽取的实体：
{entities_str}

知识库相关背景：
{kb_str}

请给出你的分析。"""

    try:
        caller = LLMCaller(llm)
        resp = caller.call(user, system=system, max_tokens=800)
        return _parse_verdict(resp.content), resp.content
    except Exception as e:
        logger.warning(f"LLM news analysis failed: {e}")
        return "unknown", f"LLM 分析失败: {e}"


def _llm_topic_assessment(llm, topic: str, items: list, combined_text: str, kb_sources: list) -> tuple:
    """LLM-powered topic research assessment — via LLMCaller."""
    from financial_rag.llm.caller import LLMCaller

    news_preview = "\n".join(
        f"  - [{item.get('source', '')}] {item.get('title', '')}"
        for item in items[:15]
    ) or "  (未获取到相关新闻)"

    kb_str = "\n".join(
        f"  - [{s['source']}] (相关度{s['score']:.2f}) {s['text'][:150]}"
        for s in kb_sources[:3]
    ) or "  (无相关知识库背景)"

    system = """你是专业的 AI/科技行业研究分析师。请对以下话题进行综合研判。

你必须输出以下格式：
【判断】利好 / 利空 / 中性（三选一）
【置信度】高 / 中 / 低
【行业动态】概括近期 2-3 个关键事件
【综合分析】200-400字，结合新闻和知识库信息，分析该话题对相关行业/公司的影响
【投资启示】1-2 句话，站在投资者角度给出建议
【风险提示】1-2 个不确定因素

要求：
- 有数据支撑、逻辑清晰、不空泛
- 基于提供的信息回答，不要编造任何数据或数字
- 如果知识库背景与话题无关（相关度低于0.5），请忽略它
- 如果信息不足，明确说明“数据不足，无法判断”
"""

    user = f"""研究话题：{topic}

近期相关新闻 ({len(items)} 条)：
{news_preview}

新闻详情摘要：
{combined_text[:3000]}

知识库相关背景：
{kb_str}

请给出综合研判。"""

    try:
        caller = LLMCaller(llm)
        resp = caller.call(user, system=system, max_tokens=1000)
        return _parse_verdict(resp.content), resp.content
    except Exception as e:
        logger.warning(f"LLM topic analysis failed: {e}")
        return "unknown", f"LLM 分析失败: {e}"


def _heuristic_assessment(doc_type: str, metrics: dict, entities: dict) -> tuple:
    """Rule-based fallback when LLM is unavailable."""
    positive = ["增长", "突破", "融资", "发布", "升级", "创新高", "超预期"]
    negative = ["下降", "亏损", "裁员", "下滑", "风险", "制裁", "暴跌"]

    lines = [f"文档类型: {doc_type}"]
    if metrics:
        lines.append(f"抽取指标: {', '.join(metrics.keys())}")
    if entities:
        companies = entities.get("companies", [])
        if companies and isinstance(companies, list) and isinstance(companies[0], dict):
            lines.append(f"涉及公司: {', '.join(c.get('name', '') for c in companies[:5])}")

    for k, v in metrics.items():
        if isinstance(v, dict):
            yoy = v.get("yoy_growth")
            if yoy and isinstance(yoy, (int, float)):
                if yoy > 20:
                    positive.append(f"{k}增长{yoy}%")
                elif yoy < -20:
                    negative.append(f"{k}下降{abs(yoy)}%")

    score = len(positive) - len(negative)
    if score > 1:
        assessment = "bullish"
    elif score < -1:
        assessment = "bearish"
    else:
        assessment = "neutral"

    lines.append(f"\n基础研判: {'利好' if assessment == 'bullish' else '利空' if assessment == 'bearish' else '中性'}")
    lines.append("(注: 无 LLM 时仅提供规则分析，配置 API Key 可获得深度分析)")
    return assessment, "\n".join(lines)
