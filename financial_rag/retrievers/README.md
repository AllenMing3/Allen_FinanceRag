# 检索模块 — 文件职责说明

## 两条主链路

**入库**：文档 → 清洗 → 切分 → Embedding → 存储
**检索**：Query → 解析 → BM25+Vector → 融合 → 精排 → 过滤 → 门控

---

## 文件清单

| 文件 | 干什么 |
|------|--------|
| `factory.py` | **装配入口**。所有零件在这组装，改配置看这里 |
| `hybrid_engine.py` | **调度中心**。编排入库和检索两条链路的执行顺序 |
| `preprocessor.py` | 入库第1步：文本清洗、相关性门控、文档类型分类 |
| `metadata.py` | 入库第2步：正则抽取 company/publish_date/sector |
| `chunker.py` | 入库第3步：按 doc_type 切分（<500字不切，长文按段落） |
| `embedding_cache.py` | 入库第4步：调 Embedding API，带本地缓存避免重复调用 |
| `vector_engine.py` | 入库第5步：向量写入 ChromaDB |
| `bm25_engine.py` | 关键词检索引擎（入库时建索引，检索时查询） |
| `query_parser.py` | 检索第1步：解析 query（关键词/日期/同义词扩展） |
| `query_planner.py` | 检索第2步：多轮查询规划（改写、扩展） |
| `fusion.py` | 检索第3步：BM25 + Vector 结果 RRF 融合 |
| `filters.py` | 检索第4步：按 metadata 过滤（日期/类型等） |
| `dictionaries.py` | 词典数据（股票/金融术语/行业/停用词） |
| `dictionary_registry.py` | 词典注册中心，统一管理多词典加载 |
| `persistence.py` | 索引持久化（save/load 到磁盘） |
| `lightrag_adapter.py` | LightRAG 图检索适配器（实验性） |

---

## 调度顺序（在 hybrid_engine.py 里）

```
入库: index() / add()
  → chunker.split_documents()
  → bm25.build()
  → embedding_cache.embed_documents()
  → vector_engine.add()

检索: search()
  → query_parser.parse()
  → bm25.search() + vector_engine.search()  (并发)
  → fusion.rrf_fusion()
  → reranker.rerank()
  → filters.apply_filters()
  → 质量门控 (rerank_score < 0.15 → 拦截)
```
