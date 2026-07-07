"""
Analysis + news + metadata + kline endpoints

Endpoints:
- GET  /api/config
- POST /api/analyze/news
- POST /api/analyze/topic
- POST /api/metadata/clear
- GET  /api/metadata/status
- POST /api/news
- POST /api/kline
"""
import asyncio
import os
import time
import logging

from fastapi import APIRouter, HTTPException

from financial_rag.api.app_state import (
    _state, _state_lock, _ensure_init,
    _META_PATH,
    _save_meta, _append_news_archive,
)
from financial_rag.api.models import (
    AnalyzeNewsRequest, AnalyzeTopicRequest, NewsRequest, KlineRequest,
    ChatFollowupRequest, CreateSessionRequest,
)
from financial_rag.api.kb_router import _save_analysis_to_kb

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple TTL cache for /api/config (rarely changes)
_config_cache: dict = {}
_config_cache_time: float = 0.0
_CONFIG_TTL = 60.0  # seconds


# ===================== Endpoints =====================


@router.get("/api/config")
async def api_config():
    global _config_cache, _config_cache_time
    await asyncio.to_thread(_ensure_init)
    now = time.time()
    if _config_cache and (now - _config_cache_time) < _CONFIG_TTL:
        return _config_cache
    cfg = _state["cfg"]
    from financial_rag.config import is_mock_enabled

    # KB health status
    kb_built = _state.get("kb_built", False)
    kb_docs = _state.get("kb_docs", [])
    init_errors = _state.get("init_errors", [])
    if kb_built:
        kb_status = {"state": "ready", "doc_count": len(kb_docs)}
    elif len(kb_docs) == 0:
        kb_status = {"state": "empty", "reason": "知识库为空，请先导入数据"}
    else:
        # Docs exist but index failed
        kb_err = next((e for e in init_errors if e["component"] == "kb_build"), None)
        reason = kb_err["error"] if kb_err else "索引构建失败"
        kb_status = {"state": "failed", "reason": reason}

    _config_cache = {
        "llm_model": cfg.llm.model,
        "embedding_model": cfg.llm.embedding_model,
        "rerank_model": cfg.llm.rerank_model,
        "provider": cfg.llm.provider,
        "has_api_key": _state["has_key"],
        "mock_mode": is_mock_enabled(),
        "kb_status": kb_status,
        "init_errors": init_errors,
    }
    _config_cache_time = now
    return _config_cache


@router.post("/api/analyze/news")
async def api_analyze_news(req: AnalyzeNewsRequest):
    """Analyze pasted news text: structured extraction + KB context + bullish/bearish verdict"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.services.analysis import analyze_news_text

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "请输入新闻内容")

    logger.info(f"[API] /analyze/news: text={len(text)}字, query={req.query!r}, kb_built={_state.get('kb_built', False)}")
    result = analyze_news_text(
        text,
        query=req.query,
        llm=_state["llm"] if _state.get("has_key") else None,
        retriever=_state.get("retriever"),
        kb_built=_state.get("kb_built", False),
    )
    logger.info(f"[API] /analyze/news: assessment={result.get('assessment')}, "
                f"analysis_type={type(result.get('analysis')).__name__}, "
                f"entities_keys={list(result.get('entities', {}).keys())}, "
                f"metrics_keys={list(result.get('metrics', {}).keys())}")

    # Auto-create conversation session
    cm = _state.get("conversation_manager")
    if cm:
        topic = req.query or ""
        if not topic:
            entities = result.get("entities", {})
            companies = entities.get("companies", [])
            if companies:
                first = companies[0]
                topic = first.get("name", "") if isinstance(first, dict) else str(first)
            if not topic:
                topic = result.get("doc_type", "") or "news"
        session_id = cm.create_session(
            session_type="news",
            title=topic[:50],
            initial_analysis=result.get("analysis", ""),
            context={
                "news_text": text,
                "structured": result.get("structured", {}),
                "metrics": result.get("metrics", {}),
                "entities": result.get("entities", {}),
                "kb_sources": result.get("kb_sources", []),
                "assessment": result.get("assessment", ""),
                "confidence": result.get("confidence", ""),
            },
        )
        result["session_id"] = session_id
    # Continuous learning: save analysis conclusion to KB
    if result.get("assessment") and result.get("analysis"):
        # Extract meaningful topic name from entities or query
        topic = req.query or ""
        if not topic:
            entities = result.get("entities", {})
            companies = entities.get("companies", [])
            if companies:
                first = companies[0]
                topic = first.get("name", "") if isinstance(first, dict) else str(first)
            if not topic:
                topic = result.get("doc_type", "") or "news"
        _save_analysis_to_kb(topic[:20], result["assessment"], str(result["analysis"]), "news",
                            confidence=result.get("confidence", ""))
        result["saved_to_kb"] = True

    return result


@router.post("/api/analyze/topic")
async def api_analyze_topic(req: AnalyzeTopicRequest):
    """Topic research: fetch news + query KB + LLM comprehensive assessment"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.services.analysis import analyze_topic_research

    topic = req.topic.strip()
    if not topic:
        raise HTTPException(400, "请输入研究话题")

    logger.info(f"[API] /analyze/topic: topic={topic!r}, max_news={req.max_news}, kb_built={_state.get('kb_built', False)}")
    result = analyze_topic_research(
        topic,
        max_news=req.max_news,
        llm=_state["llm"] if _state.get("has_key") else None,
        retriever=_state.get("retriever"),
        kb_built=_state.get("kb_built", False),
    )
    logger.info(f"[API] /analyze/topic: assessment={result.get('assessment')}, "
                f"news_count={result.get('news_count')}, kb_sources={len(result.get('kb_sources', []))}")

    # Auto-create conversation session
    cm = _state.get("conversation_manager")
    if cm:
        session_id = cm.create_session(
            session_type="topic",
            title=topic[:50],
            initial_analysis=result.get("analysis", ""),
            context={
                "topic": topic,
                "structured": result.get("structured", {}),
                "metrics": result.get("metrics", {}),
                "entities": result.get("entities", {}),
                "kb_sources": result.get("kb_sources", []),
                "assessment": result.get("assessment", ""),
                "confidence": result.get("confidence", ""),
            },
        )
        result["session_id"] = session_id
    # Continuous learning: save analysis conclusion to KB
    if result.get("assessment") and result.get("analysis"):
        _save_analysis_to_kb(topic, result["assessment"], str(result["analysis"]), "topic",
                            confidence=result.get("confidence", ""))
        result["saved_to_kb"] = True

    # Store fetched news as metadata
    from datetime import datetime
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_items = [
        {"keyword": topic, "title": n["title"], "source": n["source"],
         "publish_time": n["publish_time"], "fetched_at": fetched_at}
        for n in result.get("news", []) if n.get("title")
    ]
    if meta_items:
        with _state_lock:
            _state["meta_store"] = _state.get("meta_store", []) + meta_items
            _save_meta(_state["meta_store"])

    return result


@router.post("/api/metadata/clear")
async def api_metadata_clear():
    """Clear collected news metadata"""
    await asyncio.to_thread(_ensure_init)
    with _state_lock:
        _state["meta_store"] = []
        _save_meta([])
    return {"ok": True}


@router.get("/api/metadata/status")
async def api_metadata_status():
    """News metadata status"""
    await asyncio.to_thread(_ensure_init)
    meta = _state.get("meta_store", [])
    keywords = {}
    for m in meta:
        kw = m.get("keyword", "unknown")
        keywords[kw] = keywords.get(kw, 0) + 1
    return {
        "count": len(meta),
        "keywords": keywords,
        "path": _META_PATH,
        "file_exists": os.path.exists(_META_PATH),
    }


@router.post("/api/news")
async def api_news(req: NewsRequest):
    await asyncio.to_thread(_ensure_init)
    from financial_rag.tools.news_tools import run_news_pipeline

    data = run_news_pipeline(
        llm=_state["llm"] if _state["has_key"] else None,
        query=req.query,
        summarize=req.summarize,
        max_news=req.max_news,
    )

    # Read the generated markdown
    md_content = ""
    filepath = data.get("filepath", "")
    if filepath and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            md_content = f.read()

    # Store as metadata only — news is context/labels, NOT KB knowledge
    from datetime import datetime
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_items = []
    for item in data.get("items", []):
        title = item.get("title", "")
        if title.strip():
            meta_items.append({
                "keyword": data.get("main_keyword", ""),
                "title": title,
                "source": item.get("source", ""),
                "publish_time": item.get("publish_time", ""),
                "fetched_at": fetched_at,
            })

    with _state_lock:
        _state["meta_store"] = _state.get("meta_store", []) + meta_items
        _save_meta(_state["meta_store"])

    # Append to archive JSONL (raw data source)
    archive_path = _append_news_archive(data.get("items", []), data.get("main_keyword", ""))
    archive_count = 0
    if os.path.exists(archive_path):
        with open(archive_path, "r", encoding="utf-8") as f:
            archive_count = sum(1 for line in f if line.strip())

    return {
        "keyword": data.get("main_keyword", ""),
        "total_found": data.get("total_found", 0),
        "filepath": filepath,
        "has_summary": data.get("has_summary", False),
        "summary": data.get("summary", ""),
        "headlines": data.get("headlines", [])[:20],
        "items": data.get("items", [])[:10],
        "markdown": md_content,
        "meta_stored": len(meta_items),
        "meta_total": len(_state.get("meta_store", [])),
        "news_archive_path": archive_path,
        "news_archive_count": archive_count,
    }


@router.post("/api/kline")
async def api_kline(req: KlineRequest):
    """K线技术分析 — 按需查询，不进知识库"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.core.base import AgentContext
    from financial_rag.agents.analysis_agent import AnalysisAgent

    agent = AnalysisAgent()
    registry = _state.get("registry")
    executor = _state.get("executor")
    if registry and executor:
        agent.bind_tools(registry, executor)
    ctx = AgentContext(
        raw_input=req.query,
        metadata={
            "intent": "kline",
            "ts_code": req.ts_code,
            "name": req.name,
            "days": req.days,
            "period": req.period,
        }
    )
    result = agent.run(ctx)

    if not result.success:
        return {"success": False, "error": result.message, "query": req.query}

    data = result.data or {}
    return {
        "success": True,
        "ts_code": data.get("ts_code", ""),
        "name": data.get("name", ""),
        "days": data.get("days", req.days),
        "data_points": data.get("data_points", 0),
        "stats": data.get("stats", {}),
        "indicators": data.get("indicators", {}),
        "analysis": data.get("analysis", ""),
    }


# ===================== Conversation / Chat Endpoints =====================


@router.get("/api/chat/sessions")
async def api_chat_list_sessions():
    """List all conversation sessions"""
    await asyncio.to_thread(_ensure_init)
    cm = _state.get("conversation_manager")
    if not cm:
        raise HTTPException(500, "ConversationManager not initialized")
    return {"sessions": cm.list_sessions()}


@router.post("/api/chat/sessions")
async def api_chat_create_session(req: CreateSessionRequest):
    """Create a new conversation session"""
    await asyncio.to_thread(_ensure_init)
    cm = _state.get("conversation_manager")
    if not cm:
        raise HTTPException(500, "ConversationManager not initialized")
    session_id = cm.create_session(
        session_type=req.session_type,
        title=req.title or "新会话",
        initial_analysis=req.initial_analysis,
        context=req.context,
    )
    return {"session_id": session_id}


@router.get("/api/chat/sessions/{session_id}")
async def api_chat_get_session(session_id: str):
    """Get session details with message history"""
    await asyncio.to_thread(_ensure_init)
    cm = _state.get("conversation_manager")
    if not cm:
        raise HTTPException(500, "ConversationManager not initialized")
    session = cm.get_session(session_id)
    if not session:
        raise HTTPException(404, f"会话不存在: {session_id}")
    return session.to_dict()


@router.delete("/api/chat/sessions/{session_id}")
async def api_chat_delete_session(session_id: str):
    """Delete a conversation session"""
    await asyncio.to_thread(_ensure_init)
    cm = _state.get("conversation_manager")
    if not cm:
        raise HTTPException(500, "ConversationManager not initialized")
    if not cm.delete_session(session_id):
        raise HTTPException(404, f"会话不存在: {session_id}")
    return {"ok": True}


@router.post("/api/chat/followup")
async def api_chat_followup(req: ChatFollowupRequest):
    """Send a follow-up message and get LLM response"""
    await asyncio.to_thread(_ensure_init)
    if not _state.get("has_key"):
        raise HTTPException(400, "追问需要 DASHSCOPE_API_KEY")
    cm = _state.get("conversation_manager")
    if not cm:
        raise HTTPException(500, "ConversationManager not initialized")
    if not cm.get_session(req.session_id):
        raise HTTPException(404, f"会话不存在: {req.session_id}")

    logger.info(f"[API] /chat/followup: session={req.session_id}, message={req.message[:80]!r}")
    result = cm.followup(req.session_id, req.message, _state["llm"])
    return result
