"""
Financial RAG Web UI — FastAPI 后端

启动:  python -m financial_rag.web
访问:  http://localhost:8000
"""
import os
import sys
import time
import logging
import threading
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from financial_rag.core.scorer import ScoreGrade

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & shared singletons
# ---------------------------------------------------------------------------
app = FastAPI(title="Financial RAG", version="2.0.0")

_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Persistence layer — imported from services
from financial_rag.services.persistence import (
    KB_PATH as _KB_PATH,
    META_PATH as _META_PATH,
    NEWS_ARCHIVE_PATH as _NEWS_DB_PATH,
    load_kb as _load_kb,
    save_kb as _save_kb,
    load_meta as _load_meta,
    save_meta as _save_meta,
    append_news_archive as _append_news_archive,
)

# Lazy-init holder so import doesn't block
_state: dict = {}
_state_lock = threading.Lock()


def _ensure_init():
    """Lazy-init heavy components on first request (thread-safe)"""
    if "ready" in _state:
        return
    with _state_lock:
        # Double-check after acquiring lock
        if "ready" in _state:
            return
        from financial_rag.config import config as _cfg
        from financial_rag.llm import get_llm
        from financial_rag.core.factory import create_orchestrator, create_hybrid_retriever
        from financial_rag.core.pipeline import PipelineConfig, create_pipeline_scheduler
        from financial_rag.tools import create_financial_registry, ToolExecutor, create_tool_session
        from financial_rag.slot_filler import create_slot_filler

        _state["cfg"] = _cfg
        _state["has_key"] = bool(_cfg.llm.api_key)
        _state["llm"] = get_llm(api_key=_cfg.llm.api_key, model=_cfg.llm.model) if _state["has_key"] else None

        _state["retriever"] = create_hybrid_retriever()
        _state["registry"] = create_financial_registry(retriever=_state["retriever"], llm=_state["llm"])
        _state["executor"] = ToolExecutor(_state["registry"])
        _state["filler"] = create_slot_filler(llm=_state["llm"], verbose=False) if _state["llm"] else None
        _state["orchestrator"] = create_orchestrator(retriever=_state["retriever"], llm=_state["llm"])
        _state["scheduler"] = create_pipeline_scheduler(
            orchestrator=_state["orchestrator"],
            retriever=_state["retriever"],
            registry=_state["registry"],
            executor=_state["executor"],
            llm=_state["llm"],
            filler=_state["filler"],
            config=PipelineConfig(verbose=False),
        )
        _state["sample_docs"] = [
            {"text": "商汤科技2024年营收50.3亿元，同比增长36%，生成式AI业务收入占比达60%", "meta": {"source": "sensetime_2024"}},
            {"text": "日日新大模型API日均调用量突破2000万次，同比增长400%，企业客户数达5800家", "meta": {"source": "sensetime_2024"}},
            {"text": "训练集群规模达4万卡A100，算力利用率提升至85%，推理成本降至0.5元/百万token", "meta": {"source": "sensetime_2024"}},
            {"text": "英伟达发布Blackwell B200 GPU，单卡AI训练性能较H100提升4倍", "meta": {"source": "nvidia_2025"}},
            {"text": "智谱AI完成B+轮融资，估值超200亿元，GLM-5系列模型Q2发布", "meta": {"source": "zhipu_2025"}},
            {"text": "微软Azure部署10万张B200用于训练GPT-5，OpenAI表示推理成本将显著降低", "meta": {"source": "microsoft_2025"}},
            {"text": "谷歌宣布TPU v6将于Q4量产，直接对标Blackwell架构", "meta": {"source": "google_2025"}},
        ]
        # Load persisted KB from disk
        _state["kb_docs"] = _load_kb()
        _state["kb_built"] = False
        _state["has_samples"] = False

        # Auto-load sample data if KB is empty
        if not _state["kb_docs"]:
            _state["kb_docs"] = list(_state["sample_docs"])
            _state["has_samples"] = True
            _save_kb(_state["kb_docs"])
            logger.info(f"KB auto-loaded with {len(_state['kb_docs'])} sample docs")

        # Auto-build index for existing docs
        if _state["kb_docs"]:
            try:
                r = _state["retriever"]
                r.clear()
                r.index(_state["kb_docs"], precompute_embeddings=True)
                _state["kb_built"] = True
                logger.info(f"KB auto-built from {len(_state['kb_docs'])} docs")
            except Exception as e:
                logger.warning(f"KB auto-build failed: {e}")
        _state["kb_path"] = _KB_PATH
        _state["meta_store"] = _load_meta()

        # KB 状态摘要日志
        kb_docs = _state["kb_docs"]
        source_counts = {}
        for d in kb_docs:
            src = d.get("meta", {}).get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        kb_file_size = os.path.getsize(_KB_PATH) / 1024 if os.path.exists(_KB_PATH) else 0
        logger.info(f"[KB] 启动状态: {len(kb_docs)} 篇文档 ({kb_file_size:.1f} KB), "
                    f"{len(_state['meta_store'])} 条新闻元数据, "
                    f"来源分布: {source_counts}")

        _state["ready"] = True



@app.on_event("shutdown")
def _on_shutdown():
    """Persist state on graceful shutdown (uvicorn handles SIGINT)"""
    logger.info("Shutdown: saving state...")
    try:
        with _state_lock:
            if _state.get("kb_docs") is not None:
                _save_kb(_state["kb_docs"])
            if _state.get("meta_store") is not None:
                _save_meta(_state["meta_store"])
    except Exception as e:
        logger.warning(f"Shutdown save failed: {e}")
    else:
        logger.info("State saved.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    template: str = "quick"
    top_k: int = 5
    max_fetch: int = 10
    max_retrieve: int = 5
    verbose: bool = False


class NewsRequest(BaseModel):
    query: str
    summarize: bool = True
    max_news: int = 30


class KlineRequest(BaseModel):
    query: str
    ts_code: str = ""
    name: str = ""
    days: int = 60
    period: str = "daily"


class SlotRequest(BaseModel):
    query: str
    template: str = "quick_qa"
    top_k: int = 5
    no_freeform: bool = False


class ScoreRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestFilesRequest(BaseModel):
    dir: str = "./data/financial"
    analyze: bool = False


class IngestNewsRequest(BaseModel):
    query: str
    max_news: int = 30


class BuildRequest(BaseModel):
    documents: list = []


class AnalyzeNewsRequest(BaseModel):
    text: str
    query: str = ""


class AnalyzeTopicRequest(BaseModel):
    topic: str
    max_news: int = 20


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/config")
def api_config():
    _ensure_init()
    cfg = _state["cfg"]
    from financial_rag.config import is_mock_enabled
    return {
        "llm_model": cfg.llm.model,
        "embedding_model": cfg.llm.embedding_model,
        "rerank_model": cfg.llm.rerank_model,
        "provider": cfg.llm.provider,
        "has_api_key": _state["has_key"],
        "mock_mode": is_mock_enabled(),
    }


# ===================== KB Pipeline (Ingest → Build → Query) =====================

# Known data directories to scan
_KNOWN_DIRS = [
    ("./data/financial", "财务数据"),
    ("./data/knowledge_base", "知识库 & 新闻存档"),
]


@app.get("/api/directories")
def api_directories():
    """Scan known data directories and return file listings"""
    import json as _json

    results = []
    for dir_rel, label in _KNOWN_DIRS:
        dir_path = os.path.normpath(dir_rel)
        if not os.path.isdir(dir_path):
            results.append({"path": dir_rel, "label": label, "exists": False, "files": []})
            continue

        files = []
        total_size = 0
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            fsize = os.path.getsize(fpath)
            total_size += fsize
            # Count lines for JSONL files
            line_count = 0
            if ext == ".jsonl":
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        line_count = sum(1 for line in f if line.strip())
                except Exception:
                    pass
            # Preview first 100 chars for text files
            preview = ""
            if ext in (".txt", ".jsonl", ".json"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        preview = f.read(200).strip()
                except Exception:
                    pass
            files.append({
                "name": fname,
                "ext": ext,
                "size_kb": round(fsize / 1024, 1),
                "line_count": line_count,
                "preview": preview[:150],
            })

        results.append({
            "path": dir_rel,
            "label": label,
            "exists": True,
            "file_count": len(files),
            "total_size_kb": round(total_size / 1024, 1),
            "files": files,
        })
    return {"directories": results}


@app.post("/api/ingest/files")
def api_ingest_files(req: IngestFilesRequest):
    """Load and optionally analyze documents from a directory into the KB.

    analyze=True runs IngestionAgent + AnalysisAgent on each document
    to extract structured features (metrics, entities) before KB entry.
    """
    _ensure_init()
    import json

    dir_path = req.dir
    if not os.path.isdir(dir_path):
        raise HTTPException(400, f"目录不存在: {dir_path}")

    logger.info(f"[API] /ingest/files: dir={dir_path!r}, analyze={req.analyze}")
    t_total = time.time()

    raw_docs = []
    file_stats = []  # per-file stats for logging
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        fsize = os.path.getsize(fpath)
        doc_count_before = len(raw_docs)
        try:
            if fname.endswith(".jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        text = obj.get("content", obj.get("text", obj.get("title", "")))
                        if text:
                            orig_meta = obj.get("metadata", obj.get("meta", {}))
                            meta = {**orig_meta, "source": orig_meta.get("source", fname), "file": fpath}
                            raw_docs.append({"text": text, "meta": meta})
            elif fname.endswith(".txt"):
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        raw_docs.append({"text": text, "meta": {"source": fname, "file": fpath}})
            elif fname.endswith(".json"):
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            text = item.get("content", item.get("text", ""))
                            if text:
                                raw_docs.append({"text": text, "meta": {"source": fname}})
                    elif isinstance(data, dict):
                        text = data.get("content", data.get("text", ""))
                        if text:
                            raw_docs.append({"text": text, "meta": {"source": fname}})
        except Exception as e:
            logger.warning(f"Skip {fname}: {e}")
            continue
        doc_count = len(raw_docs) - doc_count_before
        file_stats.append((fname, fsize, doc_count))
        logger.info(f"[Ingest] 读取文件: {fname} ({fsize/1024:.1f} KB) → {doc_count} 篇文档")

    logger.info(f"[Ingest] 文件读取完成: {len(file_stats)} 个文件, {len(raw_docs)} 篇文档")

    # Optional: run agent analysis on loaded documents
    analyzed_count = 0
    if req.analyze and raw_docs:
        from financial_rag.core.base import AgentContext
        from financial_rag.agents.ingestion_agent import IngestionAgent
        from financial_rag.agents.analysis_agent import AnalysisAgent

        ingest_agent = IngestionAgent()
        extract_agent = AnalysisAgent()

        # 绑定工具能力 (从全局注册中心)
        registry = _state.get("registry")
        executor = _state.get("executor")
        if registry and executor:
            ingest_agent.bind_tools(registry, executor)
            extract_agent.bind_tools(registry, executor)

        logger.info(f"[Ingest] 开始分析 {len(raw_docs)} 篇文档...")
        for i, doc in enumerate(raw_docs, 1):
            t_doc = time.time()
            try:
                ctx = AgentContext(
                    raw_input=doc["text"][:500],
                    metadata={"news_context": _state.get("meta_store", []), "intent": "general"},
                )
                ctx.parsed_data = [doc]
                ir = ingest_agent.run(ctx)
                if ir.success and ir.context_updates:
                    for k, v in ir.context_updates.items():
                        setattr(ctx, k, v)
                er = extract_agent.run(ctx)
                if er.success and er.context_updates:
                    features = er.context_updates.get("extracted_features", {})
                    doc["meta"]["analyzed"] = True
                    doc["meta"]["metrics"] = features.get("metrics", {})
                    doc["meta"]["entities"] = features.get("entities", [])
                    analyzed_count += 1
                elapsed_doc = (time.time() - t_doc) * 1000
                src = doc["meta"].get("source", "?")
                logger.info(f"[Ingest] [{i}/{len(raw_docs)}] {src}: {elapsed_doc:.0f}ms, "
                            f"analyzed={doc['meta'].get('analyzed', False)}")
            except Exception as e:
                elapsed_doc = (time.time() - t_doc) * 1000
                logger.warning(f"[Ingest] [{i}/{len(raw_docs)}] 分析失败 ({elapsed_doc:.0f}ms): {e}")
                doc["meta"]["analyzed"] = False

    # Store in KB (thread-safe) — replace samples with real data
    with _state_lock:
        if _state.get("has_samples"):
            # Clear sample data, keep only real imports
            _state["kb_docs"] = raw_docs
            _state["has_samples"] = False
            logger.info("Sample data replaced with real imports")
        else:
            _state["kb_docs"] = _state.get("kb_docs", []) + raw_docs
        _save_kb(_state["kb_docs"])

    elapsed_total = (time.time() - t_total) * 1000
    logger.info(f"[Ingest] 完成: {len(raw_docs)} 篇文档, 分析 {analyzed_count}/{len(raw_docs)}, "
                f"总耗时 {elapsed_total:.0f}ms, KB 现有 {len(_state['kb_docs'])} 篇")
    return {
        "loaded": len(raw_docs),
        "analyzed": analyzed_count,
        "total": len(_state["kb_docs"]),
        "documents": _state["kb_docs"],
        "kb_path": _KB_PATH,
    }


@app.post("/api/ingest/news")
def api_ingest_news(req: IngestNewsRequest):
    """Fetch news and store as metadata only (NOT added to KB)"""
    _ensure_init()
    from financial_rag.tools.news_tools import run_news_pipeline

    logger.info(f"[API] /ingest/news: query={req.query!r}, max_news={req.max_news}")
    data = run_news_pipeline(
        llm=_state["llm"] if _state["has_key"] else None,
        query=req.query,
        summarize=True,
        max_news=req.max_news,
    )
    logger.info(f"[API] /ingest/news: {len(data.get('items', []))} items fetched")

    # Store as metadata only — news has no nutritional value for KB
    from datetime import datetime
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_items = []
    for item in data.get("items", []):
        meta_items.append({
            "keyword": data.get("main_keyword", ""),
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "publish_time": item.get("publish_time", ""),
            "fetched_at": fetched_at,
        })

    with _state_lock:
        _state["meta_store"] = _state.get("meta_store", []) + meta_items
        _save_meta(_state["meta_store"])

    # Also append to archive JSONL
    _append_news_archive(data.get("items", []), data.get("main_keyword", ""))

    return {
        "fetched": len(meta_items),
        "keyword": data.get("main_keyword", ""),
        "has_summary": data.get("has_summary", False),
        "summary": data.get("summary", ""),
        "headlines": data.get("headlines", [])[:10],
        "meta_total": len(_state["meta_store"]),
    }


@app.post("/api/ingest/sample")
def api_ingest_sample():
    """Load built-in sample data into KB buffer"""
    _ensure_init()
    docs = _state["sample_docs"]
    with _state_lock:
        _state["kb_docs"] = _state.get("kb_docs", []) + docs
        _save_kb(_state["kb_docs"])
    return {"loaded": len(docs), "total": len(_state["kb_docs"]), "documents": _state["kb_docs"],
            "kb_path": _KB_PATH}


@app.post("/api/build")
def api_build_kb(req: BuildRequest):
    """Build index from accumulated KB documents"""
    _ensure_init()
    documents = req.documents or _state.get("kb_docs", [])
    if not documents:
        raise HTTPException(400, "没有文档可索引，请先摄取数据")

    logger.info(f"[API] /build: {len(documents)} documents")
    r = _state["retriever"]
    r.clear()

    t0 = time.time()
    r.index(documents, precompute_embeddings=True)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"[API] /build: indexed {len(documents)} docs in {elapsed:.0f}ms")

    with _state_lock:
        _state["kb_built"] = True

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

    bm25_terms = len(r.bm25_index)
    embedding_dim = len(r.doc_embeddings[0]) if r.doc_embeddings else 0

    return {
        "doc_count": len(documents),
        "bm25_terms": bm25_terms,
        "embedding_dim": embedding_dim,
        "elapsed_ms": round(elapsed),
        "test_queries": test_queries,
        "kb_path": _KB_PATH,
    }


@app.post("/api/kb/clear")
def api_kb_clear():
    """Clear the KB: remove all docs from memory and disk"""
    _ensure_init()
    with _state_lock:
        _state["kb_docs"] = []
        _state["kb_built"] = False
        _state["retriever"].clear()
        _save_kb([])
    return {"ok": True, "kb_path": _KB_PATH}


@app.delete("/api/kb/source/{source_name}")
def api_kb_remove_source(source_name: str):
    """Remove all KB docs matching a given source name"""
    _ensure_init()
    with _state_lock:
        before = len(_state["kb_docs"])
        _state["kb_docs"] = [
            d for d in _state["kb_docs"]
            if d.get("meta", {}).get("source", "unknown") != source_name
        ]
        removed = before - len(_state["kb_docs"])
        if removed > 0:
            _save_kb(_state["kb_docs"])
            # Rebuild index if it was built
            if _state.get("kb_built") and _state["kb_docs"]:
                try:
                    _state["retriever"].clear()
                    _state["retriever"].index(_state["kb_docs"], precompute_embeddings=True)
                    logger.info(f"[KB] 索引已重建 (删除 {source_name} 后, {len(_state['kb_docs'])} 篇)")
                except Exception as e:
                    _state["kb_built"] = False
                    logger.warning(f"[KB] 索引重建失败: {e}")
            elif not _state["kb_docs"]:
                _state["retriever"].clear()
                _state["kb_built"] = False
    logger.info(f"[API] KB 删除来源: {source_name!r}, 移除 {removed} 篇, 剩余 {len(_state['kb_docs'])} 篇")
    return {"removed": removed, "remaining": len(_state["kb_docs"])}


# ===================== Intelligent Analysis (User-Facing) =====================

@app.post("/api/analyze/news")
def api_analyze_news(req: AnalyzeNewsRequest):
    """Analyze pasted news text: structured extraction + KB context + bullish/bearish verdict"""
    _ensure_init()
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
    return result


@app.post("/api/analyze/topic")
def api_analyze_topic(req: AnalyzeTopicRequest):
    """Topic research: fetch news + query KB + LLM comprehensive assessment"""
    _ensure_init()
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


@app.post("/api/metadata/clear")
def api_metadata_clear():
    """Clear collected news metadata"""
    _ensure_init()
    with _state_lock:
        _state["meta_store"] = []
        _save_meta([])
    return {"ok": True}


@app.get("/api/metadata/status")
def api_metadata_status():
    """News metadata status"""
    _ensure_init()
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


@app.get("/api/kb/status")
def api_kb_status():
    """KB status: path, doc count, built state"""
    _ensure_init()
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


@app.post("/api/kb-query")
def api_kb_query(req: QueryRequest):
    """Query against the built KB with full source citation"""
    _ensure_init()
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


@app.post("/api/pipeline")
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


@app.post("/api/news")
def api_news(req: NewsRequest):
    _ensure_init()
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


@app.post("/api/kline")
def api_kline(req: KlineRequest):
    """K线技术分析 — 按需查询，不进知识库"""
    _ensure_init()
    from financial_rag.core.base import AgentContext
    from financial_rag.agents.analysis_agent import AnalysisAgent

    agent = AnalysisAgent()
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


@app.post("/api/slot")
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


@app.post("/api/score")
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


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(_static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))
    logger.info(f"Starting Financial RAG Web UI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
