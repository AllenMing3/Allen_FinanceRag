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
from financial_rag.api.models import (
    QueryRequest, BuildRequest,
    CleanReportRequest, ChunkDemoRequest,
)

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
    """Build index from accumulated KB documents (incremental by default)"""
    await asyncio.to_thread(_ensure_init)
    documents = req.documents or _state.get("kb_docs", [])
    if not documents:
        raise HTTPException(400, "没有文档可索引，请先摄取数据")

    # Ensure all docs have doc_ids
    _assign_doc_ids(documents)

    logger.info(f"[API] /build: {len(documents)} documents (incremental)")
    r = _state["retriever"]

    t0 = time.time()
    # 增量重建：只对变化部分调 API，embedding 走缓存
    stats = r.rebuild_incremental(documents, use_chunker=False)
    elapsed = (time.time() - t0) * 1000
    logger.info(
        f"[API] /build: incremental done in {elapsed:.0f}ms "
        f"(added={stats.get('added', 0)}, removed={stats.get('removed', 0)})"
    )

    with _state_lock:
        _state["kb_built"] = True

    # Save index to disk for fast next startup
    r.save_index(_INDEX_PATH)

    bm25_terms = len(r._bm25._corpus_tokens) if r._bm25 and r._bm25._corpus_tokens else 0
    embedding_dim = len(r.doc_embeddings[0]) if r.doc_embeddings else 0

    result = {
        "doc_count": len(documents),
        "bm25_terms": bm25_terms,
        "embedding_dim": embedding_dim,
        "elapsed_ms": round(elapsed),
        "incremental": stats,
        "kb_path": _KB_PATH,
    }

    # 测试查询：可选跳过（节省 token）
    if not req.skip_test_queries:
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
        result["test_queries"] = test_queries

    return result


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


# ===================== Pipeline 白盒诊断端点 =====================


def _resolve_text(text: str, doc_index: int) -> tuple:
    """Resolve input: use text if provided, otherwise pick from KB by doc_index.
    Returns (resolved_text, source_label).
    """
    if text and text.strip():
        return text, "user_input"
    docs = _state.get("kb_docs", [])
    idx = doc_index if doc_index >= 0 else 0
    if not docs:
        raise HTTPException(400, "知识库为空且未传入 text")
    if idx >= len(docs):
        raise HTTPException(400, f"doc_index={idx} 超出范围，当前仅 {len(docs)} 篇")
    doc = docs[idx]
    return doc.get("text", ""), doc.get("meta", {}).get("source", f"kb_doc_{idx}")


@router.post("/api/pipeline/clean-report")
async def api_pipeline_clean_report(req: CleanReportRequest):
    """清洗报告: 对一段文本运行 TextPreprocessor，返回清洗前后对比 + 评分"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.retrievers.preprocessor import TextPreprocessor
    from financial_rag.core.ingestion_scorer import IngestionScoreCard

    raw_text, source = _resolve_text(req.text, req.doc_index)
    if not raw_text.strip():
        raise HTTPException(400, "解析到的文本为空")

    # Run preprocessor
    pp = TextPreprocessor()
    cleaned = pp.process(raw_text, collect_stats=True)
    stats = pp.get_last_stats()

    # Run ingestion scoring on preprocessing stage
    card = IngestionScoreCard()
    card.record_preprocessing([{"text": raw_text}])
    card.compute()
    stage = card.stages[0] if card.stages else None

    # Build logger summary
    logger.info(
        f"[Pipeline诊断/clean-report] source={source} "
        f"original={stats.original_len} cleaned={stats.cleaned_len} "
        f"retention={stats.retention:.2%} "
        f"html_removed={stats.html_removed} urls_removed={stats.urls_removed} "
        f"boilerplate={stats.boilerplate_removed} dedup={stats.paragraphs_deduped}"
    )
    if stage:
        for line in card._build_lines():
            logger.info(f"[Pipeline诊断] {line}")

    score_info = None
    if stage:
        score_info = {
            "score": round(stage.score, 3),
            "grade": stage.grade.value,
            "grade_cn": stage.grade.cn,
            "diagnosis": stage.diagnosis,
            "warnings": stage.warnings,
            "suggestions": stage.suggestions,
            "metrics": stage.metrics,
        }

    return {
        "source": source,
        "original": raw_text[:500],
        "original_len": len(raw_text),
        "cleaned": cleaned[:500],
        "cleaned_len": len(cleaned),
        "stats": {
            "html_removed": stats.html_removed,
            "urls_removed": stats.urls_removed,
            "control_removed": stats.control_removed,
            "boilerplate_removed": stats.boilerplate_removed,
            "paragraphs_deduped": stats.paragraphs_deduped,
            "retention": round(stats.retention, 4),
            "is_over_cleaned": stats.is_over_cleaned,
            "warnings": stats.warnings,
        },
        "score": score_info,
    }


@router.post("/api/pipeline/chunk-demo")
async def api_pipeline_chunk_demo(req: ChunkDemoRequest):
    """切片 + 分词 Demo: 对一段文本切片，展示前 3 个 chunk 的分词结果"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.retrievers.chunker import TextChunker
    from financial_rag.retrievers.bm25_engine import BM25Engine
    from financial_rag.retrievers.dictionaries import FINANCIAL_TERMS, INDUSTRY_TERMS
    from financial_rag.core.ingestion_scorer import IngestionScoreCard

    raw_text, source = _resolve_text(req.text, req.doc_index)
    if not raw_text.strip():
        raise HTTPException(400, "解析到的文本为空")

    # Run chunker
    chunker = TextChunker(chunk_size=req.chunk_size)
    chunks = chunker.split(raw_text, meta={"source": source})

    if not chunks:
        raise HTTPException(400, "切片结果为空")

    # Tokenize first 3 chunks
    domain_terms = FINANCIAL_TERMS | INDUSTRY_TERMS
    token_samples = []
    corpus_tokens = []
    for i, chunk in enumerate(chunks[:3]):
        tokens = BM25Engine._fallback_tokenize(chunk.get("text", ""))
        corpus_tokens.append(tokens)
        unique = set(tokens)
        matched_domain = [t for t in unique if t in domain_terms]
        token_samples.append({
            "chunk_id": chunk.get("meta", {}).get("chunk_id", i),
            "text_preview": chunk.get("text", "")[:200],
            "token_count": len(tokens),
            "unique_count": len(unique),
            "tokens": tokens[:60],  # cap output size
            "domain_terms_hit": matched_domain,
        })

    # Run ingestion scoring
    card = IngestionScoreCard()
    chunked_docs = [{"text": c.get("text", "")} for c in chunks]
    card.record_chunking(1, chunked_docs, chunk_size=req.chunk_size)
    if corpus_tokens:
        card.record_tokenization(corpus_tokens)
    card.compute()

    # Logger summary
    sizes = [len(c.get("text", "")) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    logger.info(
        f"[Pipeline诊断/chunk-demo] source={source} "
        f"text_len={len(raw_text)} chunks={len(chunks)} "
        f"avg_chunk_size={avg_size:.0f}"
    )
    for line in card._build_lines():
        logger.info(f"[Pipeline诊断] {line}")

    # Build score info
    scores = []
    for s in card.stages:
        scores.append({
            "name": s.name,
            "display": s.display,
            "score": round(s.score, 3),
            "grade": s.grade.value,
            "grade_cn": s.grade.cn,
            "diagnosis": s.diagnosis,
            "warnings": s.warnings,
            "metrics": s.metrics,
        })

    # Build chunk summary list (all chunks, brief)
    chunk_list = []
    for i, c in enumerate(chunks):
        t = c.get("text", "")
        boundary_char = t[-1] if t else ""
        chunk_list.append({
            "chunk_id": c.get("meta", {}).get("chunk_id", i),
            "size": len(t),
            "text_preview": t[:200],
            "boundary_char": boundary_char,
        })

    return {
        "source": source,
        "summary": {
            "original_len": len(raw_text),
            "chunk_count": len(chunks),
            "avg_chunk_size": round(avg_size),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
        },
        "chunks": chunk_list,
        "token_samples": token_samples,
        "scores": scores,
    }


@router.get("/api/pipeline/dict-stats")
async def api_pipeline_dict_stats():
    """词典利用率报告: 扫描 KB 文档，报告每个词典的命中情况"""
    await asyncio.to_thread(_ensure_init)
    from financial_rag.retrievers.dictionaries import (
        FINANCIAL_TERMS, INDUSTRY_TERMS, ACTION_TERMS, STOCK_MAP,
        JIEBA_FINANCE_WORDS, SYNONYM_LOOKUP, CONCEPT_MAP,
    )

    docs = _state.get("kb_docs", [])
    if not docs:
        return {
            "doc_count": 0,
            "dictionaries": {},
            "overall": {"total": 0, "matched": 0, "hit_rate": 0},
            "recommendations": ["知识库为空，无法评估词典利用率"],
        }

    # Collect all doc text (lowered) for scanning
    doc_texts = [d.get("text", "").lower() for d in docs]
    combined = " ".join(doc_texts)  # for fast substring search

    def _scan_dict(name: str, terms: set | list | dict, extract_fn=None):
        """Scan terms against doc texts. extract_fn converts dict entry to searchable string."""
        if isinstance(terms, dict):
            entries = list(terms.keys())
        elif isinstance(terms, list):
            entries = terms
        else:
            entries = list(terms)

        total = len(entries)
        matched_terms = []
        missing_terms = []
        hit_freq = 0

        for entry in entries:
            search_key = (extract_fn(entry) if extract_fn else str(entry)).lower()
            if not search_key:
                missing_terms.append(entry)
                continue
            # Count hits across all docs
            count = sum(1 for t in doc_texts if search_key in t)
            if count > 0:
                matched_terms.append(entry)
                hit_freq += count
            else:
                missing_terms.append(entry)

        return {
            "name": name,
            "total": total,
            "matched": len(matched_terms),
            "hit_rate": round(len(matched_terms) / max(total, 1), 4),
            "hit_freq": hit_freq,
            "missing_sample": [str(t) for t in missing_terms[:10]],
        }

    # Scan each dictionary
    dicts = []
    dicts.append(_scan_dict("FINANCIAL_TERMS", FINANCIAL_TERMS))
    dicts.append(_scan_dict("INDUSTRY_TERMS", INDUSTRY_TERMS))
    dicts.append(_scan_dict("ACTION_TERMS", ACTION_TERMS))
    # STOCK_MAP: keys are keywords
    dicts.append(_scan_dict("STOCK_MAP", STOCK_MAP))
    # JIEBA_FINANCE_WORDS: list of compound words
    dicts.append(_scan_dict("JIEBA_FINANCE_WORDS", JIEBA_FINANCE_WORDS))
    # SYNONYM_LOOKUP: keys are trigger terms
    dicts.append(_scan_dict("SYNONYM_LOOKUP", SYNONYM_LOOKUP))
    # CONCEPT_MAP: keys are concept triggers
    dicts.append(_scan_dict("CONCEPT_MAP", CONCEPT_MAP))

    # Overall
    total_all = sum(d["total"] for d in dicts)
    matched_all = sum(d["matched"] for d in dicts)
    overall_rate = round(matched_all / max(total_all, 1), 4)

    # Recommendations
    recs = []
    for d in dicts:
        if d["hit_rate"] < 0.1 and d["total"] > 3:
            recs.append(f"{d['name']}: 命中率仅 {d['hit_rate']:.0%}，考虑精简词典或扩充相关文档")
        elif d["hit_rate"] > 0.8:
            recs.append(f"{d['name']}: 命中率 {d['hit_rate']:.0%}，覆盖良好")
    if not recs:
        recs.append("各词典利用率均在合理范围内")

    # Logger output
    logger.info(f"[Pipeline诊断/dict-stats] KB文档数={len(docs)} 词典数={len(dicts)}")
    logger.info(f"[Pipeline诊断] {'词典名称':<24s} {'总数':>5s} {'命中':>5s} {'命中率':>7s} {'累计频次':>8s}")
    logger.info(f"[Pipeline诊断] {'-'*55}")
    for d in dicts:
        logger.info(
            f"[Pipeline诊断] {d['name']:<24s} {d['total']:>5d} {d['matched']:>5d} "
            f"{d['hit_rate']:>6.0%} {d['hit_freq']:>8d}"
        )
        if d["missing_sample"]:
            logger.info(f"[Pipeline诊断]   未命中样本: {', '.join(d['missing_sample'][:5])}")
    logger.info(f"[Pipeline诊断] 综合命中率: {overall_rate:.0%} ({matched_all}/{total_all})")
    for r in recs:
        logger.info(f"[Pipeline诊断] → {r}")

    return {
        "doc_count": len(docs),
        "dictionaries": dicts,
        "overall": {
            "total": total_all,
            "matched": matched_all,
            "hit_rate": overall_rate,
        },
        "recommendations": recs,
    }
