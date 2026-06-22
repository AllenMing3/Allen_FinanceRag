"""
Financial RAG Web UI — FastAPI 后端 (slim app wiring)

启动:  python -m financial_rag.web
访问:  http://localhost:8000

All endpoint logic lives in financial_rag/api/*_router.py modules.
This file only wires up the app, registers routers, and provides the entry point.
"""
import os
import sys
import signal
import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & static files
# ---------------------------------------------------------------------------
app = FastAPI(title="Financial RAG", version="2.0.0")

_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
from financial_rag.api.kb_router import router as _kb_router
from financial_rag.api.ingest_router import router as _ingest_router
from financial_rag.api.analysis_router import router as _analysis_router
from financial_rag.api.query_router import router as _query_router

app.include_router(_kb_router)
app.include_router(_ingest_router)
app.include_router(_analysis_router)
app.include_router(_query_router)

# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------
from financial_rag.api.app_state import _persist_state, _state  # noqa: re-exported for test_smoke


@app.on_event("shutdown")
def _on_shutdown():
    """Persist state on graceful shutdown"""
    _persist_state()


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

    def _sig_handler(sig, frame):
        logger.info(f"\n[Signal {sig}] Shutting down, saving state...")
        _persist_state()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        _persist_state()


if __name__ == "__main__":
    main()
