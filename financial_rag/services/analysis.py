"""
Financial RAG — Analysis Service

Pure business logic for news analysis and topic research.
No HTTP, no FastAPI — just functions that take dependencies and return dicts.

Functions:
    analyze_news_text  — paste news → extract + KB + LLM verdict
    analyze_topic_research — search topic → fetch news + KB + LLM verdict
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 模块级预编译正则 — 情绪关键词一次扫描替代多次 `kw in text`
_POSITIVE_KW = ["增长", "突破", "融资", "发布", "升级", "创新高", "超预期", "回暖", "获批", "盈利"]
_NEGATIVE_KW = ["下降", "亏损", "裁员", "下滑", "风险", "制裁", "暴跌", "违规", "退市", "预警"]
_POSITIVE_RE = re.compile("|".join(re.escape(k) for k in _POSITIVE_KW))
_NEGATIVE_RE = re.compile("|".join(re.escape(k) for k in _NEGATIVE_KW))


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

    # 1. Structured extraction (parallel: metrics + entities)
    doc_type = detect_document_type(text)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as _ex:
        _f_metrics = _ex.submit(extract_financial_metrics, text=text)
        _f_entities = _ex.submit(extract_entities, text=text)
        metrics = _f_metrics.result()
        entities = _f_entities.result()
    logger.info(f"[analyze_news] 抽取完成(并行): doc_type={doc_type}, "
                f"metrics_keys={list(metrics.keys())}, entities_keys={list(entities.keys())}")

    metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    entities_clean = {k: v for k, v in entities.items() if not k.startswith("_")}

    # 2. KB context retrieval
    kb_sources, kb_search_info = _search_kb(retriever, query or text[:200], kb_built)
    logger.info(f"[analyze_news] KB 检索: {len(kb_sources)} sources (raw={kb_search_info.get('total_results', 0)})")

    # 2b. 统一组装 grounding（唯一真相源：同一份同时喂 LLM 与 Guard，全程可观测）
    from financial_rag.context_builder import ContextBuilder
    kb_ctx = ContextBuilder.for_path("deep_analysis").build(kb_sources)
    kb_search_info["context_trace"] = kb_ctx.trace.to_dict()
    logger.info(
        f"[analyze_news] context 组装: 保真率={kb_ctx.trace.fidelity:.1%} "
        f"({kb_ctx.trace.kept_chars}/{kb_ctx.trace.available_chars}字) 纳入{kb_ctx.trace.included_items}项"
        + (f" 警告={kb_ctx.trace.warnings}" if kb_ctx.trace.warnings else "")
    )

    # 3. LLM assessment or rule-based fallback
    if llm:
        assessment, structured, confidence = _llm_news_assessment(llm, text, doc_type, metrics_clean, entities_clean, kb_ctx.grounding_text)
    else:
        assessment, structured = _heuristic_assessment(text, doc_type, metrics_clean, entities_clean)
        confidence = ""

    # Extract plain-text analysis from structured dict
    analysis_text = structured.get("analysis", "") if isinstance(structured, dict) else str(structured)
    logger.info(f"[analyze_news] 完成: assessment={assessment}, confidence={confidence}, analysis_len={len(analysis_text)}")
    return {
        "assessment": assessment,
        "analysis": analysis_text,
        "structured": structured if isinstance(structured, dict) else {},
        "confidence": confidence,
        "metrics": metrics_clean,
        "entities": entities_clean,
        "doc_type": doc_type,
        "kb_sources": kb_sources,
        "kb_grounding": kb_ctx.source_items,
        "kb_search_info": kb_search_info,
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
    import time as _time
    t0 = _time.time()
    from financial_rag.config import is_mock_enabled
    if is_mock_enabled():
        from financial_rag.mock_data import mock_search_news
        mock_result = mock_search_news(topic, max_news=max_news)
        items = mock_result.get("items", [])
        logger.info(f"[Mock] Topic '{topic}': {len(items)} mock news items")
    else:
        from financial_rag.tools.news_tools import run_news_pipeline
        logger.info(f"[analyze_topic] 步骤 1/4: 抓取新闻中...")
        data = run_news_pipeline(llm=llm, query=topic, summarize=False, max_news=max_news)
        items = data.get("items", [])
    logger.info(f"[analyze_topic] 步骤 1/4 完成: {len(items)} 条新闻 ({(_time.time()-t0)*1000:.0f}ms)")

    # 2. Combine news content
    t1 = _time.time()
    combined_text = "\n\n".join(
        f"{item.get('title', '')}: {item.get('content', '')}"
        for item in items if item.get("title") or item.get("content")
    )[:6000]

    # 3. KB search
    logger.info(f"[analyze_topic] 步骤 2/4: KB 检索...")
    search_query = topic
    if llm:
        try:
            from financial_rag.tools.news_tools import _extract_keywords
            keywords = _extract_keywords(llm, topic)
            search_query = keywords[0] if keywords else topic
        except Exception:
            pass
    kb_sources, kb_search_info = _search_kb(retriever, search_query, kb_built)
    logger.info(f"[analyze_topic] 步骤 2/4 完成: KB {len(kb_sources)} sources (raw={kb_search_info.get('total_results', 0)}) ({(_time.time()-t1)*1000:.0f}ms)")

    # 3b. 统一组装 grounding（唯一真相源 + 全程可观测）
    from financial_rag.context_builder import ContextBuilder
    kb_ctx = ContextBuilder.for_path("deep_analysis").build(kb_sources)
    kb_search_info["context_trace"] = kb_ctx.trace.to_dict()
    logger.info(f"[analyze_topic] context 组装: 保真率={kb_ctx.trace.fidelity:.1%} ({kb_ctx.trace.kept_chars}/{kb_ctx.trace.available_chars}字)")

    # 4. LLM assessment or fallback
    logger.info(f"[analyze_topic] 步骤 3/4: LLM 研判...")
    t2 = _time.time()
    if llm:
        assessment, structured, confidence = _llm_topic_assessment(llm, topic, items, combined_text, kb_ctx.grounding_text)
    else:
        assessment = "neutral"
        confidence = ""
        structured = {
            "verdict": "neutral",
            "confidence": "low",
            "sub_topics": [],
            "key_players": [],
            "sentiment_trend": "mixed",
            "contrarian_signals": [],
            "analysis": f"话题: {topic}\n获取新闻: {len(items)} 条\n\n配置 DASHSCOPE_API_KEY 可获得 LLM 智能研判。",
            "investment_implication": "配置 API Key 获取投资建议",
            "risks": ["无 LLM 时分析深度有限"],
        }

    analysis_text = structured.get("analysis", "") if isinstance(structured, dict) else str(structured)
    logger.info(f"[analyze_topic] 步骤 3/4 完成: assessment={assessment}, confidence={confidence} ({(_time.time()-t2)*1000:.0f}ms)")
    logger.info(f"[analyze_topic] 完成: assessment={assessment}, news_count={len(items)}, 总耗时 {(_time.time()-t0)*1000:.0f}ms")
    return {
        "assessment": assessment,
        "analysis": analysis_text,
        "structured": structured if isinstance(structured, dict) else {},
        "confidence": confidence,
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
        "kb_grounding": kb_ctx.source_items,
        "kb_search_info": kb_search_info,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _search_kb(retriever, query: str, kb_built: bool, min_score: float = 0.4) -> tuple:
    """Search KB for relevant context documents.

    Returns:
        (filtered_sources, search_info) — filtered results + diagnostics dict.
        search_info is always returned (even on empty/error) so the UI can show
        the retrieval process.
    """
    info = {
        "query": query[:200],
        "kb_built": kb_built,
        "total_results": 0,
        "above_threshold": 0,
        "top_scores": [],
        "threshold": min_score,
    }
    if not kb_built or not retriever:
        info["reason"] = "KB 未构建" if not kb_built else "retriever 为空"
        return [], info
    try:
        results, _ = retriever.search_with_scores(query, top_k=5)
        info["total_results"] = len(results)
        info["top_scores"] = [
            round(it.get("score", 0), 4) for it in results[:5]
        ]
        filtered = [
            {
                "text": it.get("text", ""),
                "source": it.get("meta", {}).get("source", ""),
                "score": round(it.get("score", 0), 4),
            }
            for it in results[:5]
            if it.get("score", 0) >= min_score
        ]
        info["above_threshold"] = len(filtered)
        if len(results) > 0:
            logger.info(f"[KB search] {len(results)} results, {len(filtered)} above threshold {min_score}")
        return filtered, info
    except Exception as e:
        logger.warning(f"KB search failed: {e}")
        info["reason"] = f"检索异常: {e}"
        return [], info


def _parse_verdict(text: str) -> str:
    """Extract bullish/bearish/neutral from LLM response text.

    Handles: 利好, 偏利好, 轻微利好, 利空, 偏利空, 看多, 看空, etc.
    Uses last-mentioned keyword for more accurate parsing in mixed-sentiment text.
    """
    bullish_keywords = ["利好", "看多", "偏多", "积极", "乐观"]
    bearish_keywords = ["利空", "看空", "偏空", "消极", "悲观"]

    # Find last occurrence of each type — final verdict usually appears last
    last_bull_pos = -1
    last_bear_pos = -1
    for kw in bullish_keywords:
        pos = text.rfind(kw)
        if pos > last_bull_pos:
            last_bull_pos = pos
    for kw in bearish_keywords:
        pos = text.rfind(kw)
        if pos > last_bear_pos:
            last_bear_pos = pos

    if last_bear_pos > last_bull_pos and last_bear_pos >= 0:
        return "bearish"
    elif last_bull_pos > last_bear_pos and last_bull_pos >= 0:
        return "bullish"
    return "neutral"


def _extract_confidence(text: str) -> str:
    """Extract confidence level from LLM response."""
    # Support both half-width colon (:), full-width colon (：), and bracket (】)
    if any(p in text for p in ["置信度】高", "置信度: 高", "置信度：高", "高置信"]):
        return "high"
    elif any(p in text for p in ["置信度】低", "置信度: 低", "置信度：低", "低置信"]):
        return "low"
    elif any(p in text for p in ["置信度】中", "置信度: 中", "置信度：中", "中置信"]):
        return "medium"
    return ""


def _text_to_structured_news(text: str, verdict: str, confidence: str) -> dict:
    """Convert free-text LLM response to structured dict (fallback parser)."""
    import re as _re
    # Extract key signals section
    signals = []
    sig_match = _re.search(r'【关键信号】(.*?)(?=【|$)', text, _re.DOTALL)
    if sig_match:
        for line in sig_match.group(1).strip().split('\n'):
            line = line.strip().lstrip('-•·')
            if line.strip():
                signals.append({"signal": line.strip()[:80], "severity": 3, "type": "neutral"})
    if not signals:
        signals.append({"signal": "见分析正文", "severity": 2, "type": "neutral"})

    # Extract risks
    risks = []
    risk_match = _re.search(r'【风险提示】(.*?)(?=【|$)', text, _re.DOTALL)
    if risk_match:
        for line in risk_match.group(1).strip().split('\n'):
            line = line.strip().lstrip('-•·')
            if line.strip():
                risks.append(line.strip()[:80])
    if not risks:
        risks.append("信息不足，需进一步观察")

    # Extract analysis section
    analysis_match = _re.search(r'【(?:综合分析|影响分析)】(.*?)(?=【|$)', text, _re.DOTALL)
    analysis_text = analysis_match.group(1).strip() if analysis_match else text

    return {
        "verdict": verdict,
        "confidence": confidence,
        "impact": {
            "industry": {"direction": verdict, "summary": "见分析正文"},
            "company": {"direction": verdict, "summary": "见分析正文"},
            "tech": {"direction": "neutral", "summary": "见分析正文"},
            "market": {"direction": verdict, "summary": "见分析正文"},
        },
        "key_signals": signals[:4],
        "analysis": analysis_text,
        "risks": risks[:2],
        "watch_next": ["持续关注后续动态"],
    }


def _text_to_structured_topic(text: str, verdict: str, confidence: str, items: list) -> dict:
    """Convert free-text LLM response to structured dict (fallback parser)."""
    import re as _re

    # Extract key events
    events = []
    event_match = _re.search(r'【行业动态】(.*?)(?=【|$)', text, _re.DOTALL)
    if event_match:
        for line in event_match.group(1).strip().split('\n'):
            line = line.strip().lstrip('-•·')
            if line.strip():
                events.append({"name": line.strip()[:50], "sentiment": "neutral", "summary": line.strip()[:80]})
    if not events:
        events.append({"name": "综合动态", "sentiment": "neutral", "summary": "见分析正文"})

    # Extract risks
    risks = []
    risk_match = _re.search(r'【风险提示】(.*?)(?=【|$)', text, _re.DOTALL)
    if risk_match:
        for line in risk_match.group(1).strip().split('\n'):
            line = line.strip().lstrip('-•·')
            if line.strip():
                risks.append(line.strip()[:80])
    if not risks:
        risks.append("信息不足，需进一步观察")

    # Extract investment implication
    invest_match = _re.search(r'【投资启示】(.*?)(?=【|$)', text, _re.DOTALL)
    investment = invest_match.group(1).strip() if invest_match else "配置 API Key 获取投资建议"

    # Extract analysis
    analysis_match = _re.search(r'【综合分析】(.*?)(?=【|$)', text, _re.DOTALL)
    analysis_text = analysis_match.group(1).strip() if analysis_match else text

    return {
        "verdict": verdict,
        "confidence": confidence,
        "sub_topics": events[:4],
        "key_players": [],
        "sentiment_trend": "mixed",
        "contrarian_signals": [],
        "analysis": analysis_text,
        "investment_implication": investment,
        "risks": risks[:2],
    }


def _llm_news_assessment(llm, text: str, doc_type: str, metrics: dict, entities: dict, kb_grounding: str) -> tuple:
    """LLM-powered news assessment with structured JSON output — via LLMCaller."""
    from financial_rag.llm.caller import LLMCaller

    import json as _json
    metrics_str = _json.dumps(metrics, ensure_ascii=False, indent=2) if metrics else "无"
    entities_str = _json.dumps(entities, ensure_ascii=False, indent=2) if entities else "无"
    # grounding 已由 ContextBuilder 统一组装（句边界截断 + 预算控制），此处直接注入
    kb_str = kb_grounding if kb_grounding else "  (无相关知识库背景)"

    from financial_rag.prompts import (
        NEWS_ASSESSMENT_SYSTEM, NEWS_ASSESSMENT_USER, NEWS_ASSESSMENT_FALLBACK_SYSTEM,
    )
    user_kwargs = dict(
        text=text[:3000], doc_type=doc_type, metrics=metrics_str,
        entities=entities_str, kb_grounding=kb_str,
    )
    user = NEWS_ASSESSMENT_USER.format(instruction="请输出 JSON 分析结果。", **user_kwargs)

    # --- 主路径: call_json 结构化输出 ---
    try:
        caller = LLMCaller(llm)
        data = caller.call_json(user, system=NEWS_ASSESSMENT_SYSTEM, max_tokens=1200, temperature=0.0)
        if data and isinstance(data, dict) and "verdict" in data:
            logger.info(f"[news_assessment] JSON 结构化成功: verdict={data.get('verdict')}, signals={len(data.get('key_signals', []))}")
            return data["verdict"], data, data.get("confidence", "")
    except Exception as e:
        logger.warning(f"[news_assessment] call_json 失败，降级到 free-text: {e}")

    # --- 降级路径: free-text 解析 ---
    try:
        caller = LLMCaller(llm)
        fallback_user = NEWS_ASSESSMENT_USER.format(instruction="请给出你的分析。", **user_kwargs)
        resp = caller.call(fallback_user, system=NEWS_ASSESSMENT_FALLBACK_SYSTEM, max_tokens=800)
        verdict = _parse_verdict(resp.content)
        confidence = _extract_confidence(resp.content)
        # 包装为结构化格式
        structured = _text_to_structured_news(resp.content, verdict, confidence)
        return verdict, structured, confidence
    except Exception as e:
        logger.warning(f"LLM news analysis failed: {e}")
        return "unknown", {"analysis": f"LLM 分析失败: {e}", "verdict": "unknown"}, ""


def _llm_topic_assessment(llm, topic: str, items: list, combined_text: str, kb_grounding: str) -> tuple:
    """LLM-powered topic research assessment with structured JSON output — via LLMCaller."""
    from financial_rag.llm.caller import LLMCaller

    news_preview = "\n".join(
        f"  - [{item.get('source', '')}] {item.get('title', '')}"
        for item in items[:15]
    ) or "  (未获取到相关新闻)"

    # grounding 已由 ContextBuilder 统一组装（句边界截断 + 预算控制），此处直接注入
    kb_str = kb_grounding if kb_grounding else "  (无相关知识库背景)"

    from financial_rag.prompts import (
        TOPIC_ASSESSMENT_SYSTEM, TOPIC_ASSESSMENT_USER, TOPIC_ASSESSMENT_FALLBACK_SYSTEM,
    )
    user_kwargs = dict(
        topic=topic, news_count=len(items), news_preview=news_preview,
        combined_text=combined_text[:3000], kb_grounding=kb_str,
    )
    user = TOPIC_ASSESSMENT_USER.format(instruction="请输出 JSON 研判结果。", **user_kwargs)

    # --- 主路径: call_json 结构化输出 ---
    try:
        caller = LLMCaller(llm)
        data = caller.call_json(user, system=TOPIC_ASSESSMENT_SYSTEM, max_tokens=1500, temperature=0.0)
        if data and isinstance(data, dict) and "verdict" in data:
            logger.info(f"[topic_assessment] JSON 结构化成功: verdict={data.get('verdict')}, sub_topics={len(data.get('sub_topics', []))}, players={len(data.get('key_players', []))}")
            return data["verdict"], data, data.get("confidence", "")
    except Exception as e:
        logger.warning(f"[topic_assessment] call_json 失败，降级到 free-text: {e}")

    # --- 降级路径: free-text 解析 ---
    try:
        caller = LLMCaller(llm)
        fallback_user = TOPIC_ASSESSMENT_USER.format(instruction="请给出综合研判。", **user_kwargs)
        resp = caller.call(fallback_user, system=TOPIC_ASSESSMENT_FALLBACK_SYSTEM, max_tokens=1000)
        verdict = _parse_verdict(resp.content)
        confidence = _extract_confidence(resp.content)
        structured = _text_to_structured_topic(resp.content, verdict, confidence, items)
        return verdict, structured, confidence
    except Exception as e:
        logger.warning(f"LLM topic analysis failed: {e}")
        return "unknown", {"analysis": f"LLM 分析失败: {e}", "verdict": "unknown"}, ""


def _heuristic_assessment(text: str, doc_type: str, metrics: dict, entities: dict) -> tuple:
    """Rule-based fallback when LLM is unavailable — returns structured dict."""
    # 用预编译正则一次扫描替代 20 次 `kw in text`
    pos_hits = len(_POSITIVE_RE.findall(text))
    neg_hits = len(_NEGATIVE_RE.findall(text))

    lines = [f"文档类型: {doc_type}"]
    companies_list = []
    if metrics:
        lines.append(f"抽取指标: {', '.join(metrics.keys())}")
    if entities:
        companies = entities.get("companies", [])
        if companies and isinstance(companies, list):
            if isinstance(companies[0], dict):
                companies_list = [c.get("name", "") for c in companies[:5]]
            else:
                companies_list = [str(c) for c in companies[:5]]
            lines.append(f"涉及公司: {', '.join(companies_list)}")

    signal_list = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            yoy = v.get("yoy_growth")
            if yoy and isinstance(yoy, (int, float)):
                if yoy > 20:
                    pos_hits += 1
                    signal_list.append({"signal": f"{k}同比增长{yoy}%", "severity": 3, "type": "positive"})
                elif yoy < -20:
                    neg_hits += 1
                    signal_list.append({"signal": f"{k}同比下降{abs(yoy)}%", "severity": 3, "type": "negative"})

    score = pos_hits - neg_hits
    if score > 1:
        assessment = "bullish"
    elif score < -1:
        assessment = "bearish"
    else:
        assessment = "neutral"

    if not signal_list:
        if assessment == "bullish":
            signal_list.append({"signal": "整体指标偏向积极", "severity": 2, "type": "positive"})
        elif assessment == "bearish":
            signal_list.append({"signal": "整体指标偏向消极", "severity": 2, "type": "negative"})
        else:
            signal_list.append({"signal": "指标信号不明确，需更多信息", "severity": 1, "type": "neutral"})

    impact = {
        "industry": {"direction": assessment, "summary": "需 LLM 深度分析"},
        "company": {"direction": assessment, "summary": ", ".join(companies_list[:2]) or "未知"},
        "tech": {"direction": "neutral", "summary": "需 LLM 深度分析"},
        "market": {"direction": assessment, "summary": "需 LLM 深度分析"},
    }

    structured = {
        "verdict": assessment,
        "confidence": "low",
        "impact": impact,
        "key_signals": signal_list,
        "analysis": "\n".join(lines) + f"\n\n基础研判: {'利好' if assessment == 'bullish' else '利空' if assessment == 'bearish' else '中性'}\n(注: 无 LLM 时仅提供规则分析，配置 API Key 可获得多维深度分析)",
        "risks": ["无 LLM 时分析深度有限"],
        "watch_next": ["配置 DASHSCOPE_API_KEY 获取深度分析"],
    }
    return assessment, structured
