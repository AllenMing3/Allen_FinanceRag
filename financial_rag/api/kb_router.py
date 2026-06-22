"""
KB management + learning endpoints

Endpoints:
- GET  /api/kb/status
- POST /api/kb-query
- POST /api/build
- POST /api/kb/clear
- DELETE /api/kb/source/{source_name}
- GET  /api/kb/search
- DELETE /api/kb/keyword/{keyword}
- GET  /api/kb/history
- GET  /api/learning/stats

Also contains _save_analysis_to_kb helper used by analysis_router.
"""
import asyncio
import os
import time
import logging

from fastapi import APIRouter, HTTPException

from financial_rag.core.scorer import ScoreGrade
from financial_rag.api.app_state import (
    _state, _state_lock, _ensure_init,
    _KB_PATH, _INDEX_PATH,
    _load_kb, _save_kb, _load_learning_history, _append_learning_record,
    _load_stats, _update_stats, _get_version,
    _assign_doc_ids, _dedup_docs,
)
from financial_rag.api.models import QueryRequest, BuildRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ===================== Internal helper =====================


def _save_analysis_to_kb(topic: str, assessment: str, analysis: str, analysis_type: str, confidence: str = ""):
    """Save analysis conclusion to KB and record in learning history.

    Also updates KB statistics (analysis counter, verdict breakdown).
    """
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Build a concise KB doc from analysis result
    text = (
        f"[{analysis_type}结论] {topic}\n"
        f"研判: {assessment}\n\n"
        f"{analysis}"
    )
    doc = {
        "text": text[:2000],  # cap to avoid bloating KB
        "meta": {
            "source": f"analysis:{analysis_type}:{topic[:20]}",
            "timestamp": now,
            "type": "analysis_result",
            "assessment": assessment,
            "confidence": confidence,
        },
    }
    with _state_lock:
        # Deduplicate: remove previous analysis for same topic+type (prefix match)
        prefix = f"analysis:{analysis_type}:{topic[:20]}"
        _state["kb_docs"] = [
            d for d in _state["kb_docs"]
            if not d.get("meta", {}).get("source", "").startswith(prefix)
        ]
        _state["kb_docs"].append(doc)
        _save_kb(_state["kb_docs"])
        # Incremental index: add single doc instead of full rebuild
        if _state.get("kb_built"):
            try:
                _state["retriever"].add([doc], use_chunker=True)
            except Exception:
                # Fallback: full rebuild if incremental fails
                try:
                    _state["retriever"].clear()
                    _state["retriever"].index(_state["kb_docs"], precompute_embeddings=True)
                    _state["retriever"].save_index(_INDEX_PATH)
                except Exception:
                    pass
    # Record in learning history (append-only log)
    record = _append_learning_record(
        topic=topic, assessment=assessment,
        analysis_type=analysis_type, confidence=confidence, kb_saved=True,
    )
    # Update stats
    _update_stats(kb_docs=_state["kb_docs"], analysis_record=record)
    logger.info(f"[KB] 分析结论已存入知识库: {analysis_type}:{topic!r} ({assessment})")


# ===================== Endpoints =====================


@router.get("/api/kb/status")
async def api_kb_status():
    """KB status: path, doc count, built state"""
    await asyncio.to_thread(_ensure_init)
    docs = _state.get("kb_docs", [])
    # Source breakdown
    sources = {}
    for d in docs:
        src = d.get("meta", {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    # Count analyzed docs
    analyzed = sum(1 for d in docs if d.get("meta", {}).get("analyzed"))
    meta_count = len(_state.get("meta_store", []))
    return {
        "kb_path": _KB_PATH,
        "doc_count": len(docs),
        "analyzed_count": analyzed,
        "meta_count": meta_count,
        "kb_built": _state.get("kb_built", False),
        "sources": sources,
        "file_exists": os.path.exists(_KB_PATH),
        "file_size_kb": round(os.path.getsize(_KB_PATH) / 1024, 1) if os.path.exists(_KB_PATH) else 0,
    }


@router.post("/api/kb-query")
async def api_kb_query(req: QueryRequest):
    """Query against the built KB with full source citation"""
    await asyncio.to_thread(_ensure_init)
    logger.info(f"[API] /kb-query: query={req.query!r}, top_k={req.top_k}")
    from financial_rag.core.scorer import PipelineScoreCard, ScoreGrade
    from financial_rag.guard.reflector import HallucinationGuard

    if not _state.get("kb_built"):
        # Auto-build if docs exist but not indexed
        docs = _state.get("kb_docs", [])
        if not docs:
            raise HTTPException(400, "知识库为空，请先导入数据")
        try:
            r = _state["retriever"]
            r.clear()
            r.index(docs, precompute_embeddings=True)
            _state["kb_built"] = True
        except Exception as e:
            raise HTTPException(500, f"自动构建索引失败: {e}")

    r = _state["retriever"]
    card = PipelineScoreCard(query=req.query)
    results, ret_card = r.search_with_scores(req.query, top_k=req.top_k)
    card.stages.extend(ret_card.stages)

    # Build retrieval list with source info and score breakdown
    retrieval = []
    for item in results[:req.top_k]:
        retrieval.append({
            "retriever": item.get("retriever", "?"),
            "score": round(item.get("score", 0), 4),
            "text": item.get("text", ""),
            "source": item.get("meta", {}).get("source", ""),
            "bm25_rank": item.get("bm25_rank"),
            "bm25_score": round(item.get("bm25_score", 0) or 0, 4),
            "vector_rank": item.get("vector_rank"),
            "vector_score": round(item.get("vector_score", 0) or 0, 4),
        })

    # Slot fill answer
    answer = ""
    fill_stats = None
    if _state["has_key"] and _state["filler"]:
        from financial_rag.templates import QUICK_QA_TEMPLATE
        filler = _state["filler"]
        context_docs = [it.get("text", "") for it in results[:req.top_k]]
        t0 = time.time()
        fill_stats_obj = filler.fill(QUICK_QA_TEMPLATE, query=req.query, context_docs=context_docs)
        answer = filler.render(QUICK_QA_TEMPLATE, fill_stats_obj)
        fill_elapsed = (time.time() - t0) * 1000
        fill_stats = {
            "filled_slots": fill_stats_obj.filled_slots,
            "total_slots": fill_stats_obj.total_slots,
            "total_tokens": fill_stats_obj.total_tokens,
            "avg_ttft_ms": round(fill_stats_obj.avg_ttft_ms),
            "parallel_gain": round(fill_stats_obj.parallel_gain * 100),
            "elapsed_ms": round(fill_elapsed),
        }

    # Scorecard
    sc = {
        "overall_score": card.overall_score(),
        "grade": ScoreGrade.from_score(card.overall_score()).value,
        "stages": [{"name": s.display_name, "score": s.score} for s in card.stages],
    }

    # Inject relevant news metadata as context
    news_context = []
    meta_store = _state.get("meta_store", [])
    if meta_store:
        query_lower = req.query.lower()
        for m in meta_store:
            kw = m.get("keyword", "").lower()
            if kw and any(p in query_lower for p in kw.split("、") if len(p) >= 2):
                news_context.append({
                    "title": m.get("title", ""),
                    "keyword": m.get("keyword", ""),
                    "publish_time": m.get("publish_time", ""),
                    "source": m.get("source", ""),
                })
        news_context = news_context[:10]

    return {
        "query": req.query,
        "answer": answer,
        "retrieval": retrieval,
        "news_context": news_context,
        "fill_stats": fill_stats,
        "scorecard": sc,
    }


@router.post("/api/build")
async def api_build_kb(req: BuildRequest):
    """Build index from accumulated KB documents"""
    await asyncio.to_thread(_ensure_init)
    documents = req.documents or _state.get("kb_docs", [])
    if not documents:
        raise HTTPException(400, "没有文档可索引，请先摄取数据")

    # Ensure all docs have doc_ids
    _assign_doc_ids(documents)

    logger.info(f"[API] /build: {len(documents)} documents")
    r = _state["retriever"]
    r.clear()

    t0 = time.time()
    r.index(documents, precompute_embeddings=True)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"[API] /build: indexed {len(documents)} docs in {elapsed:.0f}ms")

    with _state_lock:
        _state["kb_built"] = True

    # Save index to disk for fast next startup
    r.save_index(_INDEX_PATH)

    # Run test queries to verify
    test_queries = []
    for q in ["商汤科技 营收增长", "英伟达 GPU 算力"]:
        results, _ = r.search_with_scores(q, top_k=3)
        test_queries.append({
            "query": q,
            "results": [
                {"score": round(it.get("score", 0), 4), "text": it.get("text", "")[:60]}
                for it in results[:3]
            ],
        })

    bm25_terms = len(r._bm25._corpus_tokens) if r._bm25 and r._bm25._corpus_tokens else 0
    embedding_dim = len(r.doc_embeddings[0]) if r.doc_embeddings else 0

    return {
        "doc_count": len(documents),
        "bm25_terms": bm25_terms,
        "embedding_dim": embedding_dim,
        "elapsed_ms": round(elapsed),
        "test_queries": test_queries,
        "kb_path": _KB_PATH,
    }


@router.post("/api/kb/clear")
async def api_kb_clear():
    """Clear the KB: remove all docs from memory and disk"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.api.app_state import _ingest_progress
    with _state_lock:
        _state["kb_docs"] = []
        _state["kb_built"] = False
        _state["retriever"].clear()
        _save_kb([])
        # Remove stale index file
        import os as _os
        if _os.path.exists(_INDEX_PATH):
            _os.remove(_INDEX_PATH)
    # Reset background ingestion progress
    _ingest_progress.update({"running": False, "current": 0, "total": 0, "analyzed": 0, "errors": 0, "message": ""})
    _update_stats(kb_docs=[])
    return {"ok": True, "kb_path": _KB_PATH}


@router.delete("/api/kb/source/{source_name}")
async def api_kb_remove_source(source_name: str):
    """Remove all KB docs matching a given source name"""
    await asyncio.to_thread(_ensure_init)
    with _state_lock:
        # Collect indices to remove
        remove_indices = []
        keep_docs = []
        for i, d in enumerate(_state["kb_docs"]):
            if d.get("meta", {}).get("source", "unknown") == source_name:
                remove_indices.append(i)
            else:
                keep_docs.append(d)
        removed = len(remove_indices)
        if removed > 0:
            _state["kb_docs"] = keep_docs
            _save_kb(_state["kb_docs"])
            # Efficient remove: filter docs + embeddings, rebuild BM25 only
            if _state.get("kb_built") and _state["kb_docs"]:
                try:
                    _state["retriever"].remove(remove_indices)
                    _state["retriever"].save_index(_INDEX_PATH)
                    logger.info(f"[KB] 索引已更新 (删除 {source_name} 后, {len(_state['kb_docs'])} 篇)")
                except Exception as e:
                    _state["kb_built"] = False
                    logger.warning(f"[KB] 索引更新失败: {e}")
            elif not _state["kb_docs"]:
                _state["retriever"].clear()
                _state["kb_built"] = False
                import os as _os
                if _os.path.exists(_INDEX_PATH):
                    _os.remove(_INDEX_PATH)
    logger.info(f"[API] KB 删除来源: {source_name!r}, 移除 {removed} 篇, 剩余 {len(_state['kb_docs'])} 篇")
    if removed > 0:
        _update_stats(kb_docs=_state["kb_docs"])
    return {"removed": removed, "remaining": len(_state["kb_docs"])}


@router.get("/api/kb/search")
async def api_kb_search(keyword: str = "", limit: int = 50):
    """Search KB docs by keyword — preview before deletion"""
    await asyncio.to_thread(_ensure_init)
    docs = _state.get("kb_docs", [])
    logger.info(f"[API] /api/kb/search: keyword={keyword!r}, total_docs={len(docs)}")
    if not keyword:
        return {"count": len(docs), "matched": 0, "matches": [], "keyword": ""}
    kw = keyword.lower()
    matches = []
    for i, d in enumerate(docs):
        text = d.get("text", "").lower()
        source = d.get("meta", {}).get("source", "unknown")
        if kw in text or kw in source.lower():
            matches.append({
                "index": i,
                "source": source,
                "preview": d.get("text", "")[:120],
            })
            if len(matches) >= limit:
                break
    logger.info(f"[API] /api/kb/search: matched={len(matches)}")
    return {"count": len(docs), "matched": len(matches), "keyword": keyword, "matches": matches}


@router.delete("/api/kb/keyword/{keyword}")
async def api_kb_remove_keyword(keyword: str):
    """Remove all KB docs whose text contains the keyword"""
    await asyncio.to_thread(_ensure_init)
    kw = keyword.lower()
    with _state_lock:
        # Collect indices to remove
        remove_indices = []
        keep_docs = []
        for i, d in enumerate(_state["kb_docs"]):
            if kw in d.get("text", "").lower() or kw in d.get("meta", {}).get("source", "").lower():
                remove_indices.append(i)
            else:
                keep_docs.append(d)
        removed = len(remove_indices)
        if removed > 0:
            _state["kb_docs"] = keep_docs
            _save_kb(_state["kb_docs"])
            # Efficient remove: filter docs + embeddings, rebuild BM25 only
            if _state.get("kb_built") and _state["kb_docs"]:
                try:
                    _state["retriever"].remove(remove_indices)
                    _state["retriever"].save_index(_INDEX_PATH)
                    logger.info(f"[KB] 索引已更新 (删除关键词 {keyword!r} 后, {len(_state['kb_docs'])} 篇)")
                except Exception as e:
                    _state["kb_built"] = False
                    logger.warning(f"[KB] 索引更新失败: {e}")
            elif not _state["kb_docs"]:
                _state["retriever"].clear()
                _state["kb_built"] = False
                import os as _os
                if _os.path.exists(_INDEX_PATH):
                    _os.remove(_INDEX_PATH)
    logger.info(f"[API] KB 删除关键词: {keyword!r}, 移除 {removed} 篇, 剩余 {len(_state['kb_docs'])} 篇")
    if removed > 0:
        _update_stats(kb_docs=_state["kb_docs"])
    return {"removed": removed, "remaining": len(_state["kb_docs"])}


@router.get("/api/kb/history")
async def api_kb_history():
    """Return learning history from the dedicated log (newest first), plus stats summary."""
    await asyncio.to_thread(_ensure_init)
    history = _load_learning_history(limit=50)
    stats = _load_stats()
    return {"count": len(history), "history": history, "stats": stats}


@router.get("/api/learning/stats")
async def api_learning_stats():
    """KB learning statistics: analysis counts, verdict breakdown, doc counts, version."""
    await asyncio.to_thread(_ensure_init)
    stats = _load_stats()
    # Enrich with live counts
    docs = _state.get("kb_docs", [])
    stats["kb_doc_count"] = len(docs)
    stats["version"] = _get_version()
    return stats
