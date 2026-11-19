"""
Financial RAG — 数据层可观测层（打开黑盒）

纯加层、零侵入：不改任何现有引擎/persistence/router 的逻辑，
只读 introspection 回答三个问题：
1. 每个库归谁管（STORE_REGISTRY：归属模块 / 路径 / 可否再生 / 重建来源）
2. 每个操作落进哪几个库（FLOW_MAP：基于真实写入调用链 grep 得出）
3. 各库当下状态（inspect_live_state）+ 三存储一致性体检（consistency_check）

硬约束：
- 严格只读：SQLite 用 mode=ro URI 打开，不写任何文件、不触发重建，
  独立于 api/app_state._state（服务启动前也可用）。
- 路径运行时从 config 派生（monkeypatch kb_dir 后自动跟随，测试隔离不破）。
"""
import json
import logging
import os
import sqlite3
from datetime import datetime

from financial_rag.config import config
# 词典目录单一事实源：复用 registry 的常量（其本身硬编码未走 config，
# 记录在案的治理缺口）；测试通过 monkeypatch data_layer._DICT_DIR 隔离。
from financial_rag.retrievers.dictionary_registry import _DEFAULT_DICT_DIR as _DICT_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 三层定义（与 data/README.md、config 三个 @property 对齐）
# ---------------------------------------------------------------------------

LAYERS = [
    {"id": "source", "name": "真相源", "desc": "不可再生 · 务必备份 · 进 git"},
    {"id": "index", "name": "派生索引", "desc": "可删重建 · 由真相源派生 · gitignore"},
    {"id": "state", "name": "运行状态", "desc": "可删 · 计数/会话/学习记录 · gitignore"},
    {"id": "input", "name": "输入区", "desc": "人工投放待导入文件 · 代码只读不写 · 可删"},
]


def _layer_dir(layer_id: str) -> str:
    """层目录运行时派生；input 层用 config.data_dir（其余三层走 @property）。"""
    if layer_id == "input":
        return config.data_dir
    return getattr(config, f"{layer_id}_dir")

# ---------------------------------------------------------------------------
# 存储归属清单（声明式；rel 为相对所属层目录的路径，运行时拼 config 派生目录）
# kind 决定巡检计数方式；owner/writers 记录"谁在管、谁在写"
# ---------------------------------------------------------------------------

STORE_REGISTRY = [
    # —— source 真相源（不可再生）——
    {"id": "kb_docs", "name": "KB 文档", "layer": "source", "rel": "kb_docs.json",
     "kind": "json_list", "owner": "services/persistence.py",
     "writers": ["save_kb"], "regenerable": False, "rebuild_source": None,
     "desc": "知识库全部文档正文+元数据，检索的唯一真相源"},
    {"id": "news_metadata", "name": "新闻元数据", "layer": "source", "rel": "news_metadata.json",
     "kind": "json_list", "owner": "services/persistence.py",
     "writers": ["save_meta"], "regenerable": False, "rebuild_source": None,
     "desc": "新闻标题/来源/时间，供查询上下文（不进检索索引）"},
    {"id": "news_archive", "name": "新闻原始档案", "layer": "source", "rel": "news_archive.jsonl",
     "kind": "jsonl", "owner": "services/persistence.py",
     "writers": ["append_news_archive"], "regenerable": False, "rebuild_source": None,
     "desc": "JSONL 追加式原始新闻（5000 行轮转），优质正文另入 kb_docs"},
    {"id": "dictionaries", "name": "领域词典", "layer": "source", "base": "dict_dir", "rel": "",
     "kind": "dir_json", "owner": "retrievers/dictionary_registry.py",
     "writers": ["save_external", "add_words"], "regenerable": False, "rebuild_source": None,
     "desc": "jieba 外部词典 JSON（内置词表在 dictionaries.py 代码里）；运行时 add_word 注入，"
             "无独立 jieba 词典文件；写 API 无生产调用方，目前手工维护；改词典须重建 BM25"},
    # —— index 派生索引（可删重建）——
    {"id": "kb_index", "name": "索引快照", "layer": "index", "rel": "kb_index.json",
     "kind": "kb_index", "owner": "retrievers/persistence.py",
     "writers": ["save_index"], "regenerable": True, "rebuild_source": "kb_docs",
     "desc": "HybridRetriever 文档快照 v3+SHA256 校验和（权重仅构建快照，加载以代码为准）"},
    {"id": "bm25", "name": "BM25 全文索引", "layer": "index", "rel": "bm25_index.db",
     "kind": "sqlite_bm25", "owner": "retrievers/bm25_engine.py",
     "writers": ["build", "add", "remove", "clear"], "regenerable": True,
     "rebuild_source": "kb_docs",
     "desc": "SQLite FTS5 倒排索引（kb_fts 表，按 chunk 计数，WAL 模式）"},
    {"id": "chroma", "name": "Chroma 向量库", "layer": "index", "rel": "chroma",
     "kind": "sqlite_chroma", "owner": "retrievers/vector_engine.py",
     "writers": ["index", "add", "remove", "clear"], "regenerable": True,
     "rebuild_source": "kb_docs + embedding API",
     "desc": "chroma.sqlite3 + HNSW 二进制段（1024 维 cosine，按 chunk 计数）"},
    {"id": "embedding_cache", "name": "Embedding 缓存", "layer": "index", "rel": "embedding_cache.json",
     "kind": "embedding_cache", "owner": "retrievers/embedding_cache.py",
     "writers": ["embed_texts", "save"], "regenerable": True,
     "rebuild_source": "embedding API（重算费钱费时，缓存为省钱而存）",
     "desc": "文本 MD5 → 向量缓存（LRU 5000 条），避免重复调 embedding API"},
    # —— state 运行状态（可删）——
    {"id": "stats", "name": "KB 统计", "layer": "state", "rel": "stats.json",
     "kind": "json_dict", "owner": "services/persistence.py",
     "writers": ["update_stats"], "regenerable": True, "rebuild_source": "kb_docs + learning",
     "desc": "文档数/来源分布/分析计数/失败计数"},
    {"id": "kb_version", "name": "KB 版本号", "layer": "state", "rel": ".kb_version",
     "kind": "text", "owner": "services/persistence.py",
     "writers": ["_bump_version"], "regenerable": True, "rebuild_source": None,
     "desc": "每次 save_kb 自增的计数器"},
    {"id": "conversations", "name": "追问会话", "layer": "state", "rel": "conversations",
     "kind": "dir_json", "owner": "services/conversation.py",
     "writers": ["_save_session"], "regenerable": True, "rebuild_source": None,
     "desc": "每 session 一个 JSON（历史消息+压缩摘要），供开卷追问"},
    {"id": "learning", "name": "学习历史", "layer": "state", "rel": "learning/history.jsonl",
     "kind": "jsonl", "owner": "services/persistence.py",
     "writers": ["append_learning_record"], "regenerable": True, "rebuild_source": None,
     "desc": "JSONL 追加式分析结论日志（topic/研判/置信度）"},
    # —— input 输入区（kb_dir 之外，代码只读不写）——
    {"id": "financial_input", "name": "文件输入暂存", "layer": "input", "base": "data_dir", "rel": "",
     "kind": "dir_files", "owner": "core/router.py（CLI build 默认目录）",
     "writers": [], "regenerable": False, "rebuild_source": None,
     "desc": "人工投放待导入文件（config.data_dir），导入后内容流入 kb_docs，本目录可删"},
]

# ---------------------------------------------------------------------------
# 数据流图（操作 → 落库列表；由真实写入调用链 grep 得出，非臆测）
# ---------------------------------------------------------------------------

FLOW_MAP = [
    {"id": "ingest_files", "name": "文件导入", "trigger": "数据管理 → 导入所选 / 上传",
     "writes_to": ["kb_docs", "bm25", "chroma", "embedding_cache", "kb_index"],
     "note": "ingest_router：清洗去重 → save_kb → 增量索引 → save_index"},
    {"id": "ingest_news", "name": "新闻抓取", "trigger": "数据管理 → 抓取 / 深度调研自动抓取",
     "writes_to": ["news_metadata", "news_archive", "kb_docs"],
     "note": "analysis_router：标题存元数据，原始正文追加档案，优质正文另入 KB（连带索引）"},
    {"id": "save_analysis", "name": "分析结论入库", "trigger": "深度分析 → 保存到知识库",
     "writes_to": ["kb_docs", "bm25", "chroma", "embedding_cache", "kb_index", "learning", "stats"],
     "note": "kb_router._save_analysis_to_kb：结论成文入库 + 学习记录 + 统计"},
    {"id": "build_rebuild", "name": "构建/重建索引", "trigger": "数据管理 → 构建索引 / 启动失配自愈",
     "writes_to": ["bm25", "chroma", "kb_index", "embedding_cache"],
     "note": "kb_router + app_state：从 kb_docs 全量重建派生索引（真相源不动）"},
    {"id": "delete_docs", "name": "删除文档", "trigger": "数据管理 → 按来源/关键词删除",
     "writes_to": ["kb_docs", "bm25", "chroma", "kb_index", "stats"],
     "note": "kb_router：remove_by_doc_ids 同步三存储 + save_index"},
    {"id": "clear_kb", "name": "清空知识库", "trigger": "数据管理 → 清空知识库",
     "writes_to": ["kb_docs", "bm25", "chroma", "kb_index", "stats"],
     "note": "kb_router：真相源与派生索引一起清空（不可恢复，需重新导入）"},
    {"id": "chat_followup", "name": "追问对话", "trigger": "深度分析 → 追问发送",
     "writes_to": ["conversations"],
     "note": "conversation.py：每轮内存+磁盘双写会话 JSON"},
]

# ---------------------------------------------------------------------------
# 上游依赖边（读依赖，非写入）：FLOW_MAP 只画“写”，这里补“谁喂谁”
# ---------------------------------------------------------------------------

DEPENDENCIES = [
    {"from": "dictionaries", "to": "bm25",
     "note": "jieba 分词同源约束：建索引与查询必须同一套词表，词典变更后旧索引词表错位，须重建 BM25"},
    {"from": "dictionaries", "to": "kb_docs",
     "note": "入库元数据抽取：metadata.py 直读 stocks_extended.json 公司词典（绕过 DictionaryRegistry，两条独立读取路径）"},
]

# 元数据覆盖率统计字段（DocMetadata 必填 + 过滤用；正则抽取漏抽是静默的，覆盖率一查便知）
COVERAGE_FIELDS = [
    "doc_id", "title", "doc_type", "source", "date_added",
    "company", "sector", "publish_date",
]


# ---------------------------------------------------------------------------
# 路径派生（运行时读 config，保证 monkeypatch kb_dir 后跟随）
# ---------------------------------------------------------------------------

def _store_path(entry: dict) -> str:
    base = entry.get("base", "kb")
    if base == "dict_dir":
        return str(_DICT_DIR)
    if base == "data_dir":
        return os.path.join(config.data_dir, entry["rel"]) if entry["rel"] else config.data_dir
    layer_dir = getattr(config, f"{entry['layer']}_dir")
    return os.path.join(layer_dir, entry["rel"])


def _dir_size_kb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / 1024, 1)


def _fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _sqlite_count(db_path: str, table: str):
    """只读打开 SQLite 计数（mode=ro：不建 WAL、不改 mtime、锁库也不阻塞写入）。"""
    uri = "file:" + db_path.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 实时巡检（严格只读）
# ---------------------------------------------------------------------------

def _count_store(entry: dict, path: str):
    """按 kind 返回记录数；非记录型存储（stats/version）返回 None。"""
    kind = entry["kind"]
    if kind == "json_list":
        with open(path, "r", encoding="utf-8") as f:
            return len(json.load(f))
    if kind == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if kind == "kb_index":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("doc_count", len(data.get("documents", [])))
    if kind == "embedding_cache":
        with open(path, "r", encoding="utf-8") as f:
            return len(json.load(f).get("cache", {}))
    if kind == "sqlite_bm25":
        return _sqlite_count(path, "kb_fts")
    if kind == "sqlite_chroma":
        return _sqlite_count(os.path.join(path, "chroma.sqlite3"), "embeddings")
    if kind == "dir_json":
        return sum(1 for fn in os.listdir(path) if fn.endswith(".json"))
    if kind == "dir_files":
        return sum(len(files) for _r, _d, files in os.walk(path))
    # json_dict / text：非记录型，无计数语义
    return None


def inspect_live_state() -> dict:
    """只读返回每库 {exists, count, size_kb, mtime}，按 store id 索引。

    任何单库读取失败只影响该库（count=None + error），不拖垮整体巡检。
    """
    result = {}
    for entry in STORE_REGISTRY:
        path = _store_path(entry)
        info = {"exists": False, "count": None, "size_kb": 0.0, "mtime": None}
        try:
            if os.path.isdir(path):
                info["exists"] = True
                info["size_kb"] = _dir_size_kb(path)
                info["mtime"] = _fmt_mtime(os.path.getmtime(path))
                if entry["kind"] == "sqlite_chroma":
                    db = os.path.join(path, "chroma.sqlite3")
                    if not os.path.exists(db):
                        info["exists"] = False
                        result[entry["id"]] = info
                        continue
                info["count"] = _count_store(entry, path)
            elif os.path.exists(path):
                info["exists"] = True
                info["size_kb"] = round(os.path.getsize(path) / 1024, 1)
                info["mtime"] = _fmt_mtime(os.path.getmtime(path))
                info["count"] = _count_store(entry, path)
        except Exception as e:
            info["error"] = str(e)
            logger.warning(f"数据层巡检 {entry['id']} 失败: {e}")
        result[entry["id"]] = info
    return result


# ---------------------------------------------------------------------------
# 一致性体检（kb_docs / bm25 / chroma / kb_index 四方计数）
# ---------------------------------------------------------------------------

_CONSISTENCY_IDS = ["kb_docs", "bm25", "chroma", "kb_index"]


def consistency_check(live: dict = None) -> dict:
    """对比四方计数，失配/缺失标红并给原因与操作建议。

    注意：bm25/chroma 按 chunk 计数，长文切分后 > 文档数是正常现象；
    失配提示里列出全部可能原因，由人判断（体检只报事实，不自动重建）。
    """
    live = live if live is not None else inspect_live_state()
    counts = {sid: live[sid]["count"] for sid in _CONSISTENCY_IDS}
    exists = {sid: live[sid]["exists"] for sid in _CONSISTENCY_IDS}
    issues = []

    base = counts["kb_docs"]
    if not exists["kb_docs"] or not base:
        # 真相源为空：派生索引也应为空/缺失，否则是陈旧残留
        for sid in _CONSISTENCY_IDS[1:]:
            if exists[sid] and counts[sid]:
                issues.append(
                    f"{sid} 有 {counts[sid]} 条记录但真相源为空 — 派生索引陈旧，建议清空或重建"
                )
    else:
        for sid in _CONSISTENCY_IDS[1:]:
            if not exists[sid]:
                issues.append(f"{sid} 缺失 — 索引未构建，建议到「数据管理」执行构建索引")
            elif counts[sid] != base:
                issues.append(
                    f"{sid}={counts[sid]} ≠ kb_docs={base} — 可能原因：长文 chunk 切分"
                    f"（索引按 chunk 计数）、增量未同步或索引陈旧；建议重建索引后核对"
                )

    return {"ok": not issues, "counts": counts, "issues": issues}


# ---------------------------------------------------------------------------
# 元数据字段覆盖率（只读 kb_docs，统计各字段非空占比）
# ---------------------------------------------------------------------------

def metadata_coverage() -> dict:
    """文档元数据（meta 字段）覆盖率：{total, fields: {字段: {filled, pct}}}。

    文档 meta 不是独立存储，随文档存于 kb_docs/kb_index/BM25/Chroma 四份；
    这里以真相源 kb_docs 为准统计。抽取是纯正则（禁 LLM），漏抽静默，
    覆盖率低即提示词典/正则待补，不自动修数据。
    """
    path = os.path.join(config.source_dir, "kb_docs.json")
    docs = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception as e:
            logger.warning(f"元数据覆盖率读取 kb_docs 失败: {e}")
    total = len(docs)
    fields = {}
    for field in COVERAGE_FIELDS:
        filled = sum(1 for d in docs if d.get("meta", {}).get(field))
        pct = round(filled * 100.0 / total, 1) if total else 0.0
        fields[field] = {"filled": filled, "pct": pct}
    return {"total": total, "fields": fields}


# ---------------------------------------------------------------------------
# 总装（API 端点唯一入口）
# ---------------------------------------------------------------------------

def get_data_layer_map() -> dict:
    """组装 {layers, stores(清单+实时), flows, dependencies, consistency, metadata_coverage}。"""
    live = inspect_live_state()
    stores = []
    for entry in STORE_REGISTRY:
        stores.append({**entry, "path": _store_path(entry), "live": live[entry["id"]]})
    layers = [{**ly, "dir": _layer_dir(ly["id"])} for ly in LAYERS]
    return {
        "layers": layers,
        "stores": stores,
        "flows": FLOW_MAP,
        "dependencies": DEPENDENCIES,
        "consistency": consistency_check(live),
        "metadata_coverage": metadata_coverage(),
    }