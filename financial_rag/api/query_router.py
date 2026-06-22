"""
Pipeline + slot fill + scoring endpoints

Endpoints:
- POST /api/pipeline
- POST /api/slot
- POST /api/score
"""
import time
import logging

from fastapi import APIRouter, HTTPException

from financial_rag.core.scorer import ScoreGrade
from financial_rag.api.app_state import _state, _ensure_init
from financial_rag.api.models import QueryRequest, SlotRequest, ScoreRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ===================== Endpoints =====================


@router.post("/api/pipeline")
def api_pipeline(req: QueryRequest):
    _ensure_init()
    logger.info(f"[API] /pipeline: query={req.query!r}, template={req.template}, verbose={req.verbose}")
    if not _state["has_key"]:
        raise HTTPException(400, "Pipeline 需要 DASHSCOPE_API_KEY")

    from financial_rag.templates import (
        QUICK_QA_TEMPLATE, FINANCIAL_REPORT_TEMPLATE,
        NEWS_BRIEF_TEMPLATE, DEEP_ANALYSIS_TEMPLATE,
    )
    tmpl_map = {
        "quick": QUICK_QA_TEMPLATE, "fin": FINANCIAL_REPORT_TEMPLATE,
        "news": NEWS_BRIEF_TEMPLATE, "deep": DEEP_ANALYSIS_TEMPLATE,
    }
    template = tmpl_map.get(req.template, QUICK_QA_TEMPLATE)
    scheduler = _state["scheduler"]
    scheduler.config.verbose = req.verbose

    result = scheduler.run(
        query=req.query, template=template,
        max_fetch_news=req.max_fetch,
        max_retrieve=req.max_retrieve,
    )

    scorecard = None
    if result.scorecard:
        sc = result.scorecard
        scorecard = {
            "overall_score": sc.overall_score(),
            "grade": ScoreGrade.from_score(sc.overall_score()).value,
            "total_elapsed_ms": sc.total_elapsed(),
            "stages": [
                {"name": s.display_name, "score": s.score, "details": str(s.details)}
                for s in sc.stages
            ],
        }

    return {
        "query": result.query,
        "final_output": result.final_output,
        "fetch_ms": round(result.fetch_elapsed_ms),
        "index_ms": round(result.index_elapsed_ms),
        "process_ms": round(result.process_elapsed_ms),
        "output_ms": round(result.output_elapsed_ms),
        "total_ms": round(result.total_elapsed_ms),
        "errors": result.errors,
        "scorecard": scorecard,
    }


@router.post("/api/slot")
def api_slot(req: SlotRequest):
    _ensure_init()
    if not _state["has_key"]:
        raise HTTPException(400, "槽位填充需要 DASHSCOPE_API_KEY")

    from financial_rag.templates import get_template, ALL_TEMPLATES
    from financial_rag.core.scorer import PipelineScoreCard

    template = get_template(req.template)
    if not template:
        raise HTTPException(400, f"未知模板: {req.template}, 可选: {', '.join(ALL_TEMPLATES.keys())}")

    r = _state["retriever"]
    if not _state.get("kb_built"):
        docs = _state.get("kb_docs", [])
        if not docs:
            raise HTTPException(400, "知识库为空，请先导入数据")
        r.clear()
        r.index(docs, precompute_embeddings=True)
        _state["kb_built"] = True
    results, ret_card = r.search_with_scores(req.query, top_k=req.top_k)
    context_docs = [it.get("text", "") for it in results[:req.top_k]]

    # Freeform (optional)
    freeform = None
    if not req.no_freeform:
        context_text = "\n".join(doc[:200] for doc in context_docs[:3])
        t0 = time.time()
        resp = _state["llm"].chat(
            messages=f"根据以下参考信息回答问题。\n参考:\n{context_text}\n\n问题: {req.query}",
            system="你是专业金融分析师，回答必须准确有依据。不确定请说明。",
            max_tokens=600,
        )
        freeform = {
            "content": resp.content,
            "tokens": resp.usage.get("total_tokens", len(resp.content)),
            "elapsed_ms": round((time.time() - t0) * 1000),
        }

    # Slot fill
    filler = _state["filler"]
    t0 = time.time()
    fill_stats = filler.fill(template, query=req.query, context_docs=context_docs)
    rendered = filler.render(template, fill_stats)
    fill_elapsed = (time.time() - t0) * 1000

    slot_details = []
    for key, sr in fill_stats.slot_results.items():
        slot_details.append({
            "label": sr.label,
            "filled": sr.filled,
            "value": sr.value[:200],
            "ttft_ms": round(sr.ttft_ms),
        })

    return {
        "query": req.query,
        "template": template.name,
        "template_desc": template.description,
        "freeform": freeform,
        "slot_fill": {
            "rendered": rendered,
            "filled_slots": fill_stats.filled_slots,
            "total_slots": fill_stats.total_slots,
            "total_tokens": fill_stats.total_tokens,
            "avg_ttft_ms": round(fill_stats.avg_ttft_ms),
            "parallel_gain": round(fill_stats.parallel_gain * 100),
            "elapsed_ms": round(fill_elapsed),
            "slots": slot_details,
        },
    }


@router.post("/api/score")
def api_score(req: ScoreRequest):
    _ensure_init()
    r = _state["retriever"]
    if not _state.get("kb_built"):
        docs = _state.get("kb_docs", [])
        if not docs:
            raise HTTPException(400, "知识库为空，请先导入数据")
        r.clear()
        r.index(docs, precompute_embeddings=True)
        _state["kb_built"] = True
    results, card = r.search_with_scores(req.query, top_k=req.top_k)

    items = []
    for it in results:
        items.append({
            "retriever": it.get("retriever", "?"),
            "relevance": it.get("relevance_level", "?"),
            "score": round(it.get("score", 0), 4),
            "text": it.get("text", "")[:100],
        })

    stages = []
    for s in card.stages:
        stages.append({"name": s.display_name, "score": s.score, "details": str(s.details)})

    return {
        "query": req.query,
        "results": items,
        "scorecard": {
            "overall_score": card.overall_score(),
            "grade": ScoreGrade.from_score(card.overall_score()).value,
            "total_elapsed_ms": round(card.total_elapsed()),
            "stages": stages,
        },
    }
