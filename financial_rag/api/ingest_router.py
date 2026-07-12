"""
Data ingestion endpoints

Endpoints:
- GET  /api/file/preview
- GET  /api/directories
- POST /api/ingest/files
- POST /api/ingest/upload
- GET  /api/ingest/progress
- POST /api/ingest/news
"""
import asyncio
import os
import shutil
import tempfile
import time
import logging
import threading

from fastapi import APIRouter, HTTPException, UploadFile, File

from financial_rag.api.app_state import (
    _state, _state_lock, _ensure_init, _ingest_progress,
    _ingest_progress_lock,
    _KB_PATH,
    _save_kb, _save_meta, _append_news_archive,
    _assign_doc_ids, _dedup_docs,
)
from financial_rag.api.models import IngestFilesRequest, IngestNewsRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Known data directories to scan
_KNOWN_DIRS = [
    ("./data/financial", "财务数据"),
    ("./data/knowledge_base", "知识库 & 新闻存档"),
]


# ===================== Endpoints =====================


@router.get("/api/file/preview")
async def api_file_preview(path: str = "", file: str = "", lines: int = 20):
    """Preview first N lines of a file in a known directory.

    Args:
        path: directory path (must be under ./data/)
        file: filename
        lines: max lines to return (default 20, max 50)
    """
    if not path or not file:
        raise HTTPException(400, "Missing path or file")
    # Security: only allow paths under ./data/
    norm = os.path.normpath(os.path.join(path, file))
    if not norm.startswith(os.path.normpath("./data")):
        raise HTTPException(403, "Access denied: path must be under ./data/")
    if not os.path.isfile(norm):
        raise HTTPException(404, f"File not found: {file}")
    lines = min(max(1, lines), 50)
    try:
        with open(norm, "r", encoding="utf-8") as f:
            content_lines = []
            count = 0
            for line in f:
                if count >= lines:
                    break
                content_lines.append(line.rstrip())
                count += 1
        return {"file": file, "path": norm, "lines": content_lines, "truncated": count >= lines}
    except UnicodeDecodeError:
        return {"file": file, "path": norm, "lines": ["(binary file, cannot preview)"], "truncated": False}
    except Exception as e:
        raise HTTPException(500, f"Preview failed: {e}")


@router.get("/api/directories")
async def api_directories():
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
            # PDF/image files get a type label instead
            elif ext == ".pdf":
                preview = "[PDF文件]"
            elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                preview = f"[图片文件] {ext[1:].upper()}"
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


@router.post("/api/ingest/files")
async def api_ingest_files(req: IngestFilesRequest):
    """Load and optionally analyze documents from a directory into the KB.

    File reading is synchronous. Heavy LLM analysis runs in background thread.
    Poll GET /api/ingest/progress for status.
    """
    await asyncio.to_thread(_ensure_init)
    import json

    if _ingest_progress.get("running"):
        raise HTTPException(409, f"导入正在进行中: {_ingest_progress['current']}/{_ingest_progress['total']}")

    dir_path = req.dir
    if not os.path.isdir(dir_path):
        raise HTTPException(400, f"目录不存在: {dir_path}")

    logger.info(f"[API] /ingest/files: dir={dir_path!r}, analyze={req.analyze}")

    # Phase 1: Read files (fast, synchronous)
    raw_docs = []
    file_stats = []
    selected_files = set(req.files) if req.files else None
    llm = _state.get("llm")
    for fname in os.listdir(dir_path):
        # Skip files not in selection (if selection provided)
        if selected_files is not None and fname not in selected_files:
            continue
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
            elif fname.endswith(".pdf"):
                text = _parse_pdf_text(fpath)
                if text:
                    raw_docs.append({"text": text, "meta": {
                        "source": fname, "file": fpath, "parse_type": "pdf"
                    }})
            elif fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
                text = _describe_image(llm, fpath)
                if text:
                    raw_docs.append({"text": text, "meta": {
                        "source": fname, "file": fpath, "parse_type": "image"
                    }})
        except Exception as e:
            logger.warning(f"Skip {fname}: {e}")
            continue
        doc_count = len(raw_docs) - doc_count_before
        file_stats.append((fname, fsize, doc_count))
        logger.info(f"[Ingest] 读取文件: {fname} ({fsize/1024:.1f} KB) → {doc_count} 篇文档")

    logger.info(f"[Ingest] 文件读取完成: {len(file_stats)} 个文件, {len(raw_docs)} 篇文档")

    # LightRAG 图谱构建: PDF/图片解析内容送入知识图谱
    lightrag = _state.get("lightrag")
    if lightrag:
        graph_docs = [
            d for d in raw_docs
            if d.get("meta", {}).get("parse_type") in ("pdf", "image")
        ]
        if graph_docs:
            try:
                texts = [d["text"] for d in graph_docs]
                metas = [d.get("meta", {}) for d in graph_docs]
                lightrag.insert_texts(texts, metas)
                logger.info(f"[Ingest] {len(graph_docs)} 篇 PDF/图片文档已送入 LightRAG 图谱")
            except Exception as e:
                logger.warning(f"[Ingest] LightRAG 插入失败: {e}")

    # Phase 2: Assign doc_ids, dedup, then store in KB
    _assign_doc_ids(raw_docs)
    with _state_lock:
        existing = _state.get("kb_docs", [])
        new_only = _dedup_docs(existing, raw_docs)
        skipped = len(raw_docs) - len(new_only)
        if skipped > 0:
            logger.info(f"[Ingest] Skipped {skipped} duplicate docs")
        _state["kb_docs"] = existing + new_only
        _save_kb(_state["kb_docs"])
        # Incremental index: add new docs only (no full rebuild)
        if _state.get("kb_built") and new_only:
            try:
                _state["retriever"].add(new_only, use_chunker=True)
                logger.info(f"[Ingest] Incremental index: +{len(new_only)} docs")
            except Exception as e:
                logger.warning(f"[Ingest] Incremental index failed: {e}")

    # Return response based on whether analysis was requested
    actual_loaded = len(new_only)
    if actual_loaded == 0 and skipped > 0:
        return {
            "loaded": 0,
            "skipped_duplicates": skipped,
            "analyzed": 0,
            "total": len(_state["kb_docs"]),
            "status": "no_new_docs",
            "message": f"所有 {skipped} 篇文档已存在于知识库，已跳过",
            "kb_path": _KB_PATH,
        }

    # Phase 3: Launch background analysis if requested
    if req.analyze and new_only:
        _ingest_progress.update({
            "running": True, "current": 0, "total": len(new_only),
            "analyzed": 0, "errors": 0, "message": "分析中...",
        })
        t = threading.Thread(target=_run_ingest_analysis, args=(new_only,), daemon=True)
        t.start()
        return {
            "loaded": actual_loaded,
            "skipped_duplicates": skipped,
            "analyzed": 0,
            "total": len(_state["kb_docs"]),
            "status": "analyzing_in_background",
            "message": f"已导入 {actual_loaded} 篇新文档（跳过 {skipped} 篇重复），后台分析中...",
            "kb_path": _KB_PATH,
        }

    return {
        "loaded": actual_loaded,
        "skipped_duplicates": skipped,
        "analyzed": 0,
        "total": len(_state["kb_docs"]),
        "documents": [],
        "kb_path": _KB_PATH,
    }


def _parse_pdf_text(path: str) -> str:
    """解析 PDF 为纯文本（复用 document_parse_tools 的 parse_pdf_file）"""
    from financial_rag.tools.document_parse_tools import parse_pdf_file
    result = parse_pdf_file(path)
    if result.get("_error"):
        logger.warning(f"[Ingest] PDF 解析失败: {result['_error']}")
        return ""
    return result.get("text", "")


def _describe_image(llm, path: str) -> str:
    """用多模态模型解析图片（复用 document_parse_tools 的 describe_image_file）"""
    from financial_rag.tools.document_parse_tools import describe_image_file, inject_document_parse_llm
    # 确保 LLM 已注入闭包
    if llm:
        inject_document_parse_llm(llm)
    result = describe_image_file(path)
    if result.get("_error"):
        logger.warning(f"[Ingest] 图片解析失败: {result['_error']}")
        return ""
    return result.get("description", "")


def _run_ingest_analysis(raw_docs):
    """Background thread: run agent analysis on ingested documents (concurrent)."""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from financial_rag.core.base import AgentContext
        from financial_rag.agents.ingestion_agent import IngestionAgent
        from financial_rag.agents.analysis_agent import AnalysisAgent

        def _analyze_single(doc, idx, total):
            """Analyze one doc: ingest_agent + extract_agent. Returns (doc, success, elapsed_ms)."""
            # Each thread creates its own agent instances to avoid shared-state issues
            local_ingest = IngestionAgent()
            local_extract = AnalysisAgent()
            registry = _state.get("registry")
            executor_tool = _state.get("executor")
            if registry and executor_tool:
                local_ingest.bind_tools(registry, executor_tool)
                local_extract.bind_tools(registry, executor_tool)

            t_doc = time.time()
            success = False
            try:
                ctx = AgentContext(
                    raw_input=doc["text"][:500],
                    metadata={"news_context": _state.get("meta_store", []), "intent": "general"},
                )
                ctx.parsed_data = [doc]
                ir = local_ingest.run(ctx)
                if ir.success and ir.context_updates:
                    for k, v in ir.context_updates.items():
                        setattr(ctx, k, v)
                er = local_extract.run(ctx)
                if er.success and er.context_updates:
                    features = er.context_updates.get("extracted_features", {})
                    doc["meta"]["analyzed"] = True
                    doc["meta"]["metrics"] = features.get("metrics", {})
                    doc["meta"]["entities"] = features.get("entities", [])
                    success = True
                elapsed_ms = (time.time() - t_doc) * 1000
                src = doc["meta"].get("source", "?")
                logger.info(f"[Ingest-BG] [{idx}/{total}] {src}: {elapsed_ms:.0f}ms")
            except Exception as e:
                elapsed_ms = (time.time() - t_doc) * 1000
                logger.warning(f"[Ingest-BG] [{idx}/{total}] 分析失败 ({elapsed_ms:.0f}ms): {e}")
                doc["meta"]["analyzed"] = False
            return doc, success, elapsed_ms

        logger.info(f"[Ingest-BG] 开始后台分析 {len(raw_docs)} 篇文档 (并发模式)...")
        total = len(raw_docs)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_analyze_single, doc, i, total): doc
                for i, doc in enumerate(raw_docs, 1)
            }
            for future in as_completed(futures):
                try:
                    doc, success, _ = future.result()
                except Exception as e:
                    logger.warning(f"[Ingest-BG] future 异常: {e}")
                    success = False
                with _ingest_progress_lock:
                    _ingest_progress["current"] += 1
                    if success:
                        _ingest_progress["analyzed"] += 1
                    else:
                        _ingest_progress["errors"] += 1

        # Save updated KB with analysis results
        with _state_lock:
            _save_kb(_state["kb_docs"])
        _ingest_progress["message"] = f"完成: {_ingest_progress['analyzed']}/{total} 已分析"
        logger.info(f"[Ingest-BG] {_ingest_progress['message']}")
    except Exception as e:
        _ingest_progress["message"] = f"后台分析异常: {e}"
        logger.error(f"[Ingest-BG] 异常: {e}")
    finally:
        _ingest_progress["running"] = False


@router.get("/api/ingest/progress")
async def api_ingest_progress():
    """Poll background ingestion progress."""
    return dict(_ingest_progress)


@router.post("/api/ingest/upload")
async def api_ingest_upload(files: list[UploadFile] = File(...)):
    """Upload PDF/image files directly for parsing and ingestion.

    Accepted types: .pdf, .png, .jpg, .jpeg, .webp
    """
    await asyncio.to_thread(_ensure_init)

    if _ingest_progress.get("running"):
        raise HTTPException(409, f"导入正在进行中: {_ingest_progress['current']}/{_ingest_progress['total']}")

    ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    upload_dir = tempfile.mkdtemp(prefix="finrag_upload_")
    saved = []

    try:
        # Phase 1: Save uploaded files to temp dir
        for f in files:
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in ALLOWED_EXTS:
                raise HTTPException(400, f"不支持的文件类型: {f.filename}（支持 {', '.join(ALLOWED_EXTS)}）")
            dest = os.path.join(upload_dir, f.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append((f.filename, dest, ext))

        logger.info(f"[API] /ingest/upload: {len(saved)} files uploaded")

        # Phase 2: Parse files
        raw_docs = []
        llm = _state.get("llm")
        for fname, fpath, ext in saved:
            try:
                if ext == ".pdf":
                    text = _parse_pdf_text(fpath)
                    if text:
                        raw_docs.append({"text": text, "meta": {
                            "source": fname, "file": fpath, "parse_type": "pdf"
                        }})
                elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                    text = _describe_image(llm, fpath)
                    if text:
                        raw_docs.append({"text": text, "meta": {
                            "source": fname, "file": fpath, "parse_type": "image"
                        }})
            except Exception as e:
                logger.warning(f"[Upload] Skip {fname}: {e}")

        if not raw_docs:
            return {"loaded": 0, "message": "所有文件解析为空", "total": len(_state.get("kb_docs", []))}

        # LightRAG graph insertion
        lightrag = _state.get("lightrag")
        if lightrag:
            try:
                texts = [d["text"] for d in raw_docs]
                metas = [d.get("meta", {}) for d in raw_docs]
                lightrag.insert_texts(texts, metas)
                logger.info(f"[Upload] {len(raw_docs)} docs inserted into LightRAG")
            except Exception as e:
                logger.warning(f"[Upload] LightRAG insert failed: {e}")

        # Phase 3: Store in KB
        _assign_doc_ids(raw_docs)
        with _state_lock:
            existing = _state.get("kb_docs", [])
            new_only = _dedup_docs(existing, raw_docs)
            skipped = len(raw_docs) - len(new_only)
            _state["kb_docs"] = existing + new_only
            _save_kb(_state["kb_docs"])
            if _state.get("kb_built") and new_only:
                try:
                    _state["retriever"].add(new_only, use_chunker=True)
                except Exception as e:
                    logger.warning(f"[Upload] Incremental index failed: {e}")

        actual_loaded = len(new_only)
        return {
            "loaded": actual_loaded,
            "skipped_duplicates": skipped,
            "total": len(_state["kb_docs"]),
            "message": f"已导入 {actual_loaded} 篇" + (f"，跳过 {skipped} 篇重复" if skipped else ""),
        }
    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(upload_dir, ignore_errors=True)
        except Exception:
            pass


@router.post("/api/ingest/news")
async def api_ingest_news(req: IngestNewsRequest):
    """Fetch news and store as metadata only (NOT added to KB)"""
    await asyncio.to_thread(_ensure_init)
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
