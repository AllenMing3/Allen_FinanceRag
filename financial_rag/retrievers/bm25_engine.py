"""
BM25 检索引擎 — SQLite FTS5 实现

职责:
- 全文检索索引（真增量 INSERT/DELETE，无需全量重建）
- BM25 排序（FTS5 内置 bm25() 函数）
- 持久化到磁盘（.db 文件，重启不丢索引）
- 中文分词（jieba 优先，回退到 trigram）

设计:
- 文档经 jieba 分词后以空格拼接存入 FTS5
- 查询同样分词 → FTS5 MATCH → bm25() 排序
- 每次 add/delete 即时生效，无需 rebuild
- 替代原 rank_bm25 内存方案（demo 级，每次全量重建）
"""
import hashlib
import json
import logging
import os
import re
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认数据库路径
_DEFAULT_DB_DIR = os.path.join("data", "knowledge_base")
_DEFAULT_DB_NAME = "bm25_index.db"


class BM25Engine:
    """SQLite FTS5 全文检索引擎 — 真增量，持久化，BM25 排序"""

    def __init__(self, tokenizer=None, db_path: Optional[str] = None):
        """
        Args:
            tokenizer: 分词函数 callable(text) -> List[str]，None 则用回退分词
            db_path: SQLite 数据库文件路径，None 则用默认路径
        """
        self._tokenizer = tokenizer
        if db_path is None:
            os.makedirs(_DEFAULT_DB_DIR, exist_ok=True)
            db_path = os.path.join(_DEFAULT_DB_DIR, _DEFAULT_DB_NAME)
        self._db_path = db_path
        self._initialized = False
        self._ensure_table()

    # ===================== 初始化 =====================

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（每次操作独立连接，线程安全）"""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        """确保 FTS5 表存在"""
        if self._initialized:
            return
        conn = self._get_conn()
        try:
            # FTS5 虚拟表：tokenized 用于 MATCH，doc_json 存完整文档
            # tokenize="unicode61" 对已分词文本（空格分隔）按空格切分即可
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
                    tokenized,
                    doc_json UNINDEXED,
                    doc_key UNINDEXED,
                    tokenize="unicode61"
                )
            """)
            conn.commit()
            self._initialized = True
        except Exception as e:
            logger.error(f"FTS5 表创建失败: {e}")
            raise
        finally:
            conn.close()

    # ===================== 索引操作 =====================

    def build(self, documents: List[Dict]):
        """全量重建索引（清空 + 批量写入）

        用于: 首次加载、load_index、数据不一致时的兜底重建。
        日常增量请用 add() / remove()。
        """
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM kb_fts")
            self._insert_batch(conn, documents)
            conn.commit()
            logger.debug(f"BM25 FTS5: 全量重建 {len(documents)} 篇文档")
        finally:
            conn.close()

    def add(self, documents: List[Dict]):
        """增量添加文档（即时生效，无需重建）"""
        if not documents:
            return
        conn = self._get_conn()
        try:
            self._insert_batch(conn, documents)
            conn.commit()
            logger.debug(f"BM25 FTS5: 增量添加 {len(documents)} 篇")
        finally:
            conn.close()

    def remove(self, doc_keys: List[str]):
        """按 doc_key 删除文档（即时生效）"""
        if not doc_keys:
            return
        conn = self._get_conn()
        try:
            conn.executemany(
                "DELETE FROM kb_fts WHERE doc_key = ?",
                [(k,) for k in doc_keys]
            )
            conn.commit()
            logger.debug(f"BM25 FTS5: 删除 {len(doc_keys)} 篇")
        finally:
            conn.close()

    def remove_by_docs(self, documents: List[Dict]):
        """按文档列表删除（自动计算 doc_key）"""
        keys = [self.doc_key(d) for d in documents]
        self.remove(keys)

    def clear(self):
        """清空索引"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM kb_fts")
            conn.commit()
            logger.debug("BM25 FTS5: 索引已清空")
        finally:
            conn.close()

    # ===================== 检索 =====================

    def search(self, documents: List[Dict], query: str, top_k: int,
               query_tokens: List[str] = None) -> List[Dict]:
        """BM25 全文检索

        Args:
            documents: 文档列表（保留接口兼容，FTS5 内部已有数据）
            query: 查询文本
            top_k: 返回数量
            query_tokens: 预分词的查询 terms（可选）

        Returns:
            [{"text": ..., "meta": ..., "score": float, "retriever": "bm25", "rank": int}]
        """
        terms = query_tokens if query_tokens is not None else self.tokenize(query)
        if not terms:
            return []

        match_expr = self._build_match_expr(terms)
        if not match_expr:
            return []

        conn = self._get_conn()
        try:
            # bm25() 返回负数（越小越相关），取负转正
            rows = conn.execute(
                """SELECT doc_json, -bm25(kb_fts) as score
                   FROM kb_fts
                   WHERE kb_fts MATCH ?
                   ORDER BY bm25(kb_fts)
                   LIMIT ?""",
                (match_expr, top_k)
            ).fetchall()
        except Exception as e:
            logger.warning(f"FTS5 查询失败，回退空结果: {e}")
            return []
        finally:
            conn.close()

        results = []
        for rank, (doc_json, score) in enumerate(rows, 1):
            try:
                doc = json.loads(doc_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if score <= 0:
                continue
            results.append({
                **doc,
                "score": float(score),
                "retriever": "bm25",
                "rank": rank,
            })
        return results

    # ===================== 分词 =====================

    def tokenize(self, text: str) -> List[str]:
        """分词 — 优先注入的分词器，否则回退到正则"""
        if self._tokenizer is not None:
            try:
                return self._tokenizer(text)
            except Exception as e:
                logger.warning(f"BM25Engine: tokenizer failed, fallback regex: {e}")
        return self._fallback_tokenize(text)

    # ===================== 属性 / 兼容 =====================

    @property
    def is_built(self) -> bool:
        """索引中是否有文档"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM kb_fts").fetchone()
            return row[0] > 0
        except Exception:
            return False
        finally:
            conn.close()

    @property
    def doc_count(self) -> int:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM kb_fts").fetchone()
            return row[0]
        except Exception:
            return 0
        finally:
            conn.close()

    def get_corpus_tokens(self) -> List[List[str]]:
        """获取所有文档的分词结果（供 IngestionScoreCard 使用）"""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT tokenized FROM kb_fts").fetchall()
            return [row[0].split() for row in rows]
        except Exception:
            return []
        finally:
            conn.close()

    @property
    def _corpus_tokens(self) -> List[List[str]]:
        """兼容旧接口（kb_router / scorecard 直接访问）"""
        return self.get_corpus_tokens()

    # ===================== 内部工具 =====================

    @staticmethod
    def doc_key(doc: Dict) -> str:
        """生成文档唯一标识（优先用 meta.doc_id，否则 hash）"""
        meta = doc.get("meta", {})
        if meta.get("doc_id"):
            return str(meta["doc_id"])
        text = doc.get("text", "")[:200]
        source = meta.get("source", "unknown")
        return hashlib.md5(f"{source}|{text}".encode("utf-8", errors="replace")).hexdigest()[:16]

    def _insert_batch(self, conn: sqlite3.Connection, documents: List[Dict]):
        """批量插入文档到 FTS5"""
        rows = []
        for doc in documents:
            text = doc.get("text", "")
            tokens = self.tokenize(text)
            tokenized = " ".join(tokens)
            doc_json = json.dumps(doc, ensure_ascii=False)
            key = self.doc_key(doc)
            rows.append((tokenized, doc_json, key))

        # 去重：同 doc_key 只保留最后一条
        seen = {}
        for i, (_, _, key) in enumerate(rows):
            seen[key] = i
        if len(seen) < len(rows):
            keep = sorted(seen.values())
            rows = [rows[i] for i in keep]
            logger.debug(f"BM25 FTS5: 批内去重 {len(rows)} → {len(seen)}")

        conn.executemany(
            "INSERT INTO kb_fts(tokenized, doc_json, doc_key) VALUES (?, ?, ?)",
            rows
        )

    @staticmethod
    def _build_match_expr(terms: List[str]) -> str:
        """构建 FTS5 MATCH 表达式

        策略: 各 term 用 OR 连接（匹配任一即召回），bm25() 自动按命中数排序。
        每个 term 加双引号防止特殊字符干扰 MATCH 语法。
        重复 term 保留（FTS5 bm25 会计算 query term frequency 作为权重）。
        """
        safe = []
        for t in terms:
            # 去掉双引号防止注入，保留其他字符
            t_clean = t.replace('"', '').strip()
            if t_clean:
                safe.append(f'"{t_clean}"')
        if not safe:
            return ""
        return " OR ".join(safe)

    @staticmethod
    def _fallback_tokenize(text: str) -> List[str]:
        """回退分词: 中文 trigram + 完整段，英文按单词

        改进: 用 trigram(3字)替代 bigram(2字)，减少无意义碎片 token。
        例: "营业收入" → ["营业收", "业收入", "营业收入"]
        """
        raw = re.findall(
            r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+(?:\.\d+)?[%\uff05]?', text.lower()
        )
        tokens = []
        for seg in raw:
            if re.match(r'^[\u4e00-\u9fff]+$', seg):
                # 中文段: 始终保留完整段 (<= 8字)
                if 2 <= len(seg) <= 8:
                    tokens.append(seg)
                # >= 4字: 生成 trigram 滑窗，提升匹配精度
                if len(seg) >= 4:
                    for j in range(len(seg) - 2):
                        tokens.append(seg[j:j + 3])
                # 2-3字: 已经作为完整段保留，不再生成子串
            else:
                tokens.append(seg)
        return tokens
