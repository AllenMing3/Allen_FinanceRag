"""
Shared application state — lazy-init, persistence, shutdown handling

All router modules import _state, _state_lock, _ingest_progress, _ensure_init
from this module to share the same singleton state.
"""
import os
import logging
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence layer — imported from services
# ---------------------------------------------------------------------------
from financial_rag.services.persistence import (
    KB_PATH as _KB_PATH,
    META_PATH as _META_PATH,
    NEWS_ARCHIVE_PATH as _NEWS_DB_PATH,
    INDEX_PATH as _INDEX_PATH,
    load_kb as _load_kb,
    save_kb as _save_kb,
    load_meta as _load_meta,
    save_meta as _save_meta,
    append_news_archive as _append_news_archive,
    append_learning_record as _append_learning_record,
    load_learning_history as _load_learning_history,
    load_stats as _load_stats,
    update_stats as _update_stats,
    get_version as _get_version,
    assign_doc_ids as _assign_doc_ids,
    dedup_docs as _dedup_docs,
)

# Lazy-init holder so import doesn't block
_state: dict = {}
_state_lock = threading.Lock()

# Background ingestion progress tracker
_ingest_progress: dict = {
    "running": False,
    "current": 0,
    "total": 0,
    "analyzed": 0,
    "errors": 0,
    "message": "",
}


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
        _state["kb_docs"] = _load_kb()
        _state["kb_built"] = False

        # Assign stable doc_ids to all loaded docs (idempotent)
        _assign_doc_ids(_state["kb_docs"])

        # Auto-build or load persisted index
        if _state["kb_docs"]:
            try:
                r = _state["retriever"]
                r.clear()
                # Try loading saved index (avoids expensive re-embedding)
                import os as _os
                if _os.path.exists(_INDEX_PATH):
                    try:
                        r.load_index(_INDEX_PATH)
                        # Validate: index doc count must match KB doc count
                        if len(r.documents) == len(_state["kb_docs"]):
                            _state["kb_built"] = True
                            logger.info(f"KB index loaded from disk ({len(r.documents)} docs, no re-embed needed)")
                        else:
                            # Stale index — rebuild
                            r.clear()
                            raise ValueError("doc count mismatch, rebuilding")
                    except Exception as e:
                        logger.info(f"Saved index not usable ({e}), rebuilding...")
                        r.clear()

                if not _state["kb_built"]:
                    r.index(_state["kb_docs"], precompute_embeddings=True)
                    _state["kb_built"] = True
                    # Persist index for next startup
                    r.save_index(_INDEX_PATH)
                    logger.info(f"KB built & index saved ({len(_state['kb_docs'])} docs)")
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

        # Refresh stats on startup to reflect current KB state
        _update_stats(kb_docs=_state["kb_docs"])

        _state["ready"] = True


def _persist_state():
    """Save KB + metadata + index to disk. Safe to call multiple times."""
    try:
        with _state_lock:
            if _state.get("kb_docs") is not None:
                _save_kb(_state["kb_docs"])
            if _state.get("meta_store") is not None:
                _save_meta(_state["meta_store"])
            # Persist index for fast next startup
            if _state.get("kb_built") and _state.get("retriever"):
                try:
                    _state["retriever"].save_index(_INDEX_PATH)
                except Exception as e:
                    logger.warning(f"[Shutdown] Index save failed: {e}")
        logger.info("[Shutdown] State saved.")
    except Exception as e:
        logger.warning(f"[Shutdown] Save failed: {e}")
