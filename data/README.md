# data/ 数据目录说明

本目录存放 FinRAG 的全部持久化数据。核心设计：**按"数据能否再生"分三层**。
所有路径统一由 `financial_rag/config.py` 的 `kb_dir` 派生（`source_dir` / `index_dir` / `state_dir`），
改 `kb_dir` 一处即可整体迁移，不再有分散的硬编码路径。

## 顶层结构

| 目录 | 用途 | 进 git |
|---|---|---|
| `dictionaries/` | 只读领域词典（AI 域名词、公司信息、股票扩展），检索时做关键词扩展/过滤 | 按需 |
| `financial/` | 输入样本（待分析的原始文本，如 `test_article.txt`） | 按需 |
| `knowledge_base/` | 知识库主体，下分 `source/` `index/` `state/` 三层 | 见下 |

## knowledge_base/ 三层

### `source/` — 真相源层（不可再生 · **进 git** · 务必备份）

唯一权威数据，删了无法自动恢复。备份只需备份这一层。

| 文件 | 内容 | 写入方 |
|---|---|---|
| `kb_docs.json` | 知识库全部文档 `[{text, meta}]` | `save_kb()` 整表覆写 + 版本号 +1 |
| `news_metadata.json` | 新闻元数据（关键词/标题/来源/时间），仅作上下文标签、非 KB 知识 | `save_meta()` |
| `news_archive.jsonl` | 原始新闻归档（append-only，超 5000 行滚动截断保留最新） | `append_news_archive()` |

### `index/` — 派生索引层（**可整个删除，重启自动重建** · 不进 git）

全部由 `source/` 派生。删除后重启服务会从 `kb_docs.json` 重建
（注意：`embedding_cache.json` 重建需重新调 embedding API，有成本，非必要不删）。

| 文件/目录 | 内容 | 重建来源 |
|---|---|---|
| `kb_index.json` | 检索器文档索引 + 配置 + checksum | `retriever.save_index()` |
| `bm25_index.db` | BM25 全文检索（SQLite FTS5） | `BM25Engine.build()` |
| `chroma/` | 向量库（ChromaDB PersistentClient） | `VectorEngine.index()` |
| `embedding_cache.json` | 文本 hash → embedding 向量缓存（省重复 API 调用） | 增量累积 |
| `lightrag/` | LightRAG 知识图谱（默认关闭，仅 PDF/图片走图谱） | 图谱构建 |

### `state/` — 运行状态层（**可删** · 不进 git）

运行时统计与会话，删除不影响知识库数据本身。

| 文件/目录 | 内容 |
|---|---|
| `stats.json` | KB 统计（文档数/来源分布/分析计数/版本），启动时按 `source/` 刷新 |
| `.kb_version` | KB 版本计数器（每次 `save_kb` +1） |
| `conversations/` | 会话历史（每会话一个 json） |
| `learning/history.jsonl` | 学习记录（分析历史 append-only） |

## 常见操作

- **彻底重建索引**：删除整个 `index/` → 重启服务 → 从 `source/kb_docs.json` 自动重建。
- **安全备份**：只需备份 `source/`（`index/` `state/` 均可再生）。
- **整体迁移数据目录**：改 `config.kb_dir`（或环境变量）一处即可，三层路径自动跟随。

## 备份机制

`persistence._atomic_write_json()` 采用"写临时文件 → 原子 rename"避免写坏，
并在覆写前保留**单代 `.bak`**。所有 `.bak` / `.bak_pre_clean` 已在 `.gitignore` 忽略。
