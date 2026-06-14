"""
Financial RAG — Persistence Layer

KB documents (JSON), news metadata (JSON), news archive (JSONL).
All paths configurable; defaults to data/knowledge_base/.
"""
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base"))

KB_PATH = os.path.join(_BASE, "kb_docs.json")
META_PATH = os.path.join(_BASE, "news_metadata.json")
NEWS_ARCHIVE_PATH = os.path.join(_BASE, "news_archive.jsonl")

os.makedirs(_BASE, exist_ok=True)


# ---------------------------------------------------------------------------
# KB documents
# ---------------------------------------------------------------------------

def load_kb(path: str = KB_PATH) -> list:
    """Load KB documents from disk (or empty list)"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            logger.info(f"KB loaded: {len(docs)} docs from {path}")
            return docs
        except Exception as e:
            logger.warning(f"Failed to load KB: {e}")
    return []


def save_kb(docs: list, path: str = KB_PATH):
    """Persist KB documents to disk"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info(f"KB saved: {len(docs)} docs → {path}")


# ---------------------------------------------------------------------------
# News metadata
# ---------------------------------------------------------------------------

def load_meta(path: str = META_PATH) -> list:
    """Load news metadata from disk (or empty list)"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_meta(meta: list, path: str = META_PATH):
    """Persist news metadata to disk"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved: {len(meta)} items → {path}")


# ---------------------------------------------------------------------------
# News archive (JSONL — append-only raw data source)
# ---------------------------------------------------------------------------

def append_news_archive(items: list, keyword: str, path: str = NEWS_ARCHIVE_PATH) -> str:
    """Append raw news items to the JSONL archive file, return path"""
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            if not text.strip():
                continue
            record = {
                "text": text.strip(),
                "metadata": {
                    "source": "news",
                    "keyword": keyword,
                    "title": item.get("title", ""),
                    "publish_time": item.get("publish_time", ""),
                    "content_url": item.get("content_url", ""),
                    "fetched_at": fetched_at,
                    "doc_type": "新闻报道",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count > 0:
        logger.info(f"News archive: appended {count} items → {path}")
    return path
