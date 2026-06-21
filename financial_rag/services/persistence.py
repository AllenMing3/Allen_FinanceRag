"""
Financial RAG — Persistence Layer

KB documents (JSON), news metadata (JSON), news archive (JSONL),
learning history (JSONL), KB statistics (JSON), version tracking.

All paths configurable; defaults to data/knowledge_base/.
"""
import os
import json
import hashlib
import logging
import shutil
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base"))
_LEARNING_DIR = os.path.join(_BASE, "learning")

KB_PATH = os.path.join(_BASE, "kb_docs.json")
META_PATH = os.path.join(_BASE, "news_metadata.json")
NEWS_ARCHIVE_PATH = os.path.join(_BASE, "news_archive.jsonl")
STATS_PATH = os.path.join(_BASE, "stats.json")
VERSION_PATH = os.path.join(_BASE, ".kb_version")
LEARNING_HISTORY_PATH = os.path.join(_LEARNING_DIR, "history.jsonl")

# Index persistence path (for HybridRetriever save/load)
INDEX_PATH = os.path.join(_BASE, "kb_index.json")

# News archive rotation cap (max lines kept after trim)
NEWS_ARCHIVE_MAX_LINES = 5000

os.makedirs(_BASE, exist_ok=True)
os.makedirs(_LEARNING_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data, ensure_ascii: bool = False, indent: int = 2, backup: bool = True):
    """Write JSON atomically: write to .tmp then rename to avoid corruption.

    If backup=True and target file exists, keeps a .bak copy of the previous version.
    """
    # Create .bak backup of existing file (single-generation)
    if backup and os.path.exists(path):
        bak_path = path + ".bak"
        try:
            shutil.copy2(path, bak_path)
        except OSError as e:
            logger.warning(f"Backup copy failed: {e}")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    # Atomic rename (on same filesystem)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Version tracking
# ---------------------------------------------------------------------------

def _bump_version(path: str = VERSION_PATH) -> int:
    """Increment and return the KB version counter."""
    version = 0
    if os.path.exists(path):
        try:
            version = int(open(path, "r").read().strip())
        except (ValueError, OSError):
            pass
    version += 1
    with open(path, "w") as f:
        f.write(str(version))
    return version


def get_version(path: str = VERSION_PATH) -> int:
    """Read current KB version."""
    if os.path.exists(path):
        try:
            return int(open(path, "r").read().strip())
        except (ValueError, OSError):
            pass
    return 0


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
    """Persist KB documents to disk (atomic write + version bump)"""
    _atomic_write_json(path, docs)
    version = _bump_version()
    logger.info(f"KB saved: {len(docs)} docs → {path} (v{version})")


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
    """Persist news metadata to disk (atomic write)"""
    _atomic_write_json(path, meta)
    logger.info(f"Metadata saved: {len(meta)} items → {path}")


# ---------------------------------------------------------------------------
# News archive (JSONL — append-only raw data source)
# ---------------------------------------------------------------------------

def append_news_archive(items: list, keyword: str, path: str = NEWS_ARCHIVE_PATH,
                        max_lines: int = NEWS_ARCHIVE_MAX_LINES) -> str:
    """Append raw news items to the JSONL archive file, return path.

    After appending, rotates the file to keep at most `max_lines` lines
    (removes oldest entries from the top).
    """
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

    # Rotate: keep only the last max_lines lines
    _rotate_jsonl(path, max_lines)
    return path


def _rotate_jsonl(path: str, max_lines: int):
    """Trim a JSONL file to at most max_lines lines (keep newest / tail)."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= max_lines:
            return
        # Keep tail (newest entries)
        trimmed = lines[-max_lines:]
        removed = len(lines) - max_lines
        tmp_path = path + ".rotating"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(trimmed)
        os.replace(tmp_path, path)
        logger.info(f"News archive rotated: removed {removed} oldest lines, kept {max_lines}")
    except Exception as e:
        logger.warning(f"News archive rotation failed: {e}")


# ---------------------------------------------------------------------------
# Learning Store (append-only JSONL log of analysis history)
# ---------------------------------------------------------------------------

def append_learning_record(
    topic: str,
    assessment: str,
    analysis_type: str,
    confidence: str = "",
    kb_saved: bool = True,
    path: str = LEARNING_HISTORY_PATH,
) -> dict:
    """Append a learning record to the history log.

    Returns the record dict for caller use.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": now,
        "topic": topic,
        "assessment": assessment,
        "analysis_type": analysis_type,
        "confidence": confidence,
        "kb_saved": kb_saved,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Learning record: {analysis_type}/{topic} → {assessment} (saved={kb_saved})")
    return record


def load_learning_history(path: str = LEARNING_HISTORY_PATH, limit: int = 50) -> list:
    """Load recent learning records from history log (newest first)."""
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to load learning history: {e}")
    return list(reversed(records[-limit:]))


# ---------------------------------------------------------------------------
# KB Statistics
# ---------------------------------------------------------------------------

def load_stats(path: str = STATS_PATH) -> dict:
    """Load KB statistics from disk."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return _default_stats()


def _default_stats() -> dict:
    """Return default empty stats structure."""
    return {
        "kb_doc_count": 0,
        "kb_sources": {},
        "total_saves": 0,
        "total_analyses": 0,
        "analyses_by_type": {},
        "analyses_by_verdict": {},
        "llm_failures": 0,
        "first_save": None,
        "last_save": None,
        "version": 0,
    }


def update_stats(kb_docs: list = None, analysis_record: dict = None,
                 llm_failed: bool = False, path: str = STATS_PATH) -> dict:
    """Update and persist KB statistics.

    Args:
        kb_docs: if provided, recompute doc counts and sources
        analysis_record: if provided, increment analysis counters
        llm_failed: if True, increment failure counter
    """
    stats = load_stats(path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Update KB doc counts
    if kb_docs is not None:
        stats["kb_doc_count"] = len(kb_docs)
        source_counts = {}
        for d in kb_docs:
            src = d.get("meta", {}).get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        stats["kb_sources"] = source_counts
        stats["total_saves"] = stats.get("total_saves", 0) + 1
        if not stats.get("first_save"):
            stats["first_save"] = now
        stats["last_save"] = now

    # Update analysis counters
    if analysis_record:
        stats["total_analyses"] = stats.get("total_analyses", 0) + 1
        atype = analysis_record.get("analysis_type", "unknown")
        by_type = stats.get("analyses_by_type", {})
        by_type[atype] = by_type.get(atype, 0) + 1
        stats["analyses_by_type"] = by_type

        verdict = analysis_record.get("assessment", "unknown")
        by_verdict = stats.get("analyses_by_verdict", {})
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        stats["analyses_by_verdict"] = by_verdict

    if llm_failed:
        stats["llm_failures"] = stats.get("llm_failures", 0) + 1

    stats["version"] = get_version()
    _atomic_write_json(path, stats)
    logger.info(f"Stats updated: {stats['kb_doc_count']} docs, {stats['total_analyses']} analyses, v{stats['version']}")
    return stats


# ---------------------------------------------------------------------------
# Document identity & deduplication
# ---------------------------------------------------------------------------

def make_doc_id(doc: dict) -> str:
    """Generate a stable doc_id from source + first 200 chars of text.

    Uses SHA-256 truncated to 16 hex chars for compactness.
    """
    source = doc.get("meta", {}).get("source", "unknown")
    text = doc.get("text", "")[:200]
    key = f"{source}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def assign_doc_ids(docs: list) -> list:
    """Ensure every doc has a 'doc_id' in its meta. Mutates in place."""
    for d in docs:
        meta = d.setdefault("meta", {})
        if not meta.get("doc_id"):
            meta["doc_id"] = make_doc_id(d)
    return docs


def dedup_docs(existing: list, new_docs: list) -> list:
    """Return only new_docs whose doc_id is NOT already in existing.

    Both lists must have doc_ids assigned (call assign_doc_ids first).
    """
    seen = {d.get("meta", {}).get("doc_id") for d in existing if d.get("meta", {}).get("doc_id")}
    added = []
    for d in new_docs:
        did = d.get("meta", {}).get("doc_id")
        if did and did in seen:
            continue
        added.append(d)
        if did:
            seen.add(did)
    return added
