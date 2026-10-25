# FinRAG 架构基准

> **本文件是项目唯一架构文档**（2026-10 重写，取代已删除的 ARCHITECTURE_CN / AGENT_TOOLS / PROJECT_QA / USER_GUIDE）。
> 纪律：**代码即真理**。本文与代码冲突时以代码为准，并立即修正本文。任何调度层/通路改动必须同步本文件。

---

## 1. 项目定位

面向 AI/科技行业的金融智能分析系统（意图感知多智能体 RAG）。
核心业务：**新闻解读**（最高优先级）、话题调研、K线技术分析、知识库问答。
技术栈：FastAPI + DashScope(Qwen) + ChromaDB + 自研 HybridRetriever + LightRAG 图谱。

## 2. 总体架构

```
┌─ 前端 (static/) ─────────────────────────────────────────────┐
│  4 Tab: 系统概览 | 数据管理 | 智能查询 | 深度分析              │
│  index.html + app.js + modules/(9个ES模块)                   │
└──────────────────────────┬───────────────────────────────────┘
┌─ API 层 (api/) ──────────▼───────────────────────────────────┐
│  analysis_router | kb_router | ingest_router | query_router  │
│  app_state: 全局单例 (_state: llm/retriever/registry/        │
│             orchestrator/scheduler/filler/conversation_mgr)  │
└──────────────────────────┬───────────────────────────────────┘
┌─ 调度层 (core/) ─────────▼───────────────────────────────────┐
│  AgentRouter     意图分类(关键词+正则) → Agent 链             │
│  PipelineScheduler  5阶段: Fetch→Index→Process→Output→Evolve │
│  AgentOrchestrator  顺序执行链 + AgentContext 数据流          │
└──────────────────────────┬───────────────────────────────────┘
┌─ Agent 层 (agents/) ─────▼───────────────────────────────────┐
│  IngestionAgent → AnalysisAgent → ScoringAgent   (共3个)     │
│  Agent 只做编排，重活全部 call_tool() 委托                    │
└──────────────────────────┬───────────────────────────────────┘
┌─ 工具层 (tools/) ────────▼───────────────────────────────────┐
│  FunctionRegistry: 37 tools / 11 模块 / 5 分类               │
└──────────────────────────┬───────────────────────────────────┘
┌─ 基础设施 ───────────────▼───────────────────────────────────┐
│  retrievers/ 混合检索(BM25 FTS5+Chroma+RRF+Rerank+扩词)      │
│  llm/ DashScope客户端+ModelRouter   guard/ 6层防幻觉          │
│  services/ 分析·会话·持久化          slot_filler+templates    │
└──────────────────────────────────────────────────────────────┘
```

## 3. 业务通路（P0 — 一切工作围绕这些通路）

### 通路 A：新闻解读 ⭐ 最高优先级
```
POST /api/analyze/news (analysis_router.py)
  → _run_analysis_via_agent_chain(intent="news", parsed_data=[{text}])
    → orchestrator.set_pipeline(["AnalysisAgent","ScoringAgent"]).execute()
      → AnalysisAgent._run_deep_news_chain → call_tool("analyze_news_deep")
        (内部: KB检索 grounding + LLM 结构化分析 → verdict/key_signals/impact/risks/watch_next)
      → ScoringAgent → evaluate_pipeline_quality ‖ check_hallucination → generate_score_report
  → 链失败 fallback: services/analysis.analyze_news_text + 独立 guard 校验
  → 后处理: 创建会话(conversation_manager) + 分析结论存入 KB
```

### 通路 B：智能查询（双模式）
```
KB问答:   POST /api/kb-query (kb_router.py)
  → retriever.search 混合检索 → LLM 生成带 [N] 引用答案
  → PipelineScoreCard + HallucinationGuard(mode=rag) 直连（不走 Agent 链）
深度研究: POST /api/pipeline (query_router.py)
  → scheduler.run() 5阶段:
    1 Fetch   FC会话选数据工具拉新闻 (需 LLM)
    2 Index   TextPreprocessor 清洗 → retriever 入库+检索
    3 Process AgentRouter.route(query) → 动态链 → orchestrator.execute()
    4 Output  SlotFiller 槽位填充（Agent 已有输出则跳过）
    5 Evolve  ScoreCard 打分 + Guard 防幻觉（报告追加到输出尾部）
```

### 通路 C：话题调研
```
POST /api/analyze/topic → _run_analysis_via_agent_chain(intent="deep_topic")
  → AnalysisAgent._run_deep_topic_chain → call_tool("analyze_topic_deep")
  ⚠️ deep_topic 不在 AgentRouter 意图表里，由 API 直接注入（历史旁路）
```

### 通路 D：数据摄取
```
新闻:  POST /api/ingest/news → tools/news_tools.run_news_pipeline
       (RSS/同花顺/新浪/东财 → 元数据 news_metadata.json；优质新闻入 KB)
文件:  POST /api/ingest/files | /api/ingest/upload (拖拽上传)
       → IngestionAgent: JSONL/TXT/PDF(PyMuPDF)/图片(qwen-vl) →
         清洗 → 相关性门控 → 分类 → extract_document_metadata → KB
       → PDF/图片解析内容另送 LightRAG 图谱抽取实体关系
```

### 通路 E：K线技术分析
```
POST /api/kline → 直接实例化 AnalysisAgent (intent=kline)
  → analyze_kline(Tushare/模拟数据 + MACD/RSI/KDJ/BOLL) → generate_kline_analysis(LLM)
  ⚠️ 第三条编排入口（不经 orchestrator/不评分）
```

### 意图 → Agent 链映射（AgentRouter，core/agent_router.py）
| intent | 触发 | 链 |
|--------|------|-----|
| kline | 走势/MACD/涨跌… | Analysis → Scoring |
| event_impact | 利好/利空/大事… | Analysis → Scoring |
| report | 财报/营收/净利润… | Ingestion → Analysis → Scoring |
| news | 新闻/资讯/公告… | Ingestion → Analysis → Scoring |
| general | 兜底 (conf=0.5) | Ingestion → Analysis → Scoring |
| deep_topic | ⚠️ 仅 API 注入 | Analysis → Scoring |

链的真实区分度只有"是否带 Ingestion"；行为差异由 `metadata.intent` 驱动
AnalysisAgent 内部工具链分发（`_run_kline_chain` / `_run_event_chain` /
`_run_deep_news_chain` / `_run_deep_topic_chain` / `_run_extraction_chain`）。

## 4. 调度层执行语义

- **AgentContext**：raw_input / parsed_data / extracted_features /
  intermediate_findings / final_answer / metadata，Agent 间唯一数据通道。
- **context_updates 合并规则**（orchestrator._apply_updates）：
  dict 字段 merge、list 字段 extend、标量覆盖；未知 key 落入 metadata。
- **ScoringAgent 通用接口**：上游经 metadata 传
  `scoring_source_items` / `scoring_mode`("rag"|"analysis") / `scoring_text` 三字段即可接入评分。
- **防幻觉**：guard/reflector.HallucinationGuard，6 层（4 规则层 + LLM critique + LLM assist），
  RAG 模式查 `[N]` 引用，analysis 模式查 `【】` 结构。

## 5. Agent 清单（3 个，agents/）

| Agent | 职责 | 调用的工具 |
|-------|------|-----------|
| IngestionAgent | 文件/文本/目录摄取，清洗+门控+分类+元数据 | extract_document_metadata, detect_document_type, parse_pdf_file, describe_image_file |
| AnalysisAgent | 按 intent 分发 5 条工具链，渲染 Markdown | analyze_kline, generate_kline_analysis, fetch_date_events ‖ fetch_kline_context, assess_event_impact, analyze_news_deep, analyze_topic_deep, extract_financial_metrics ‖ extract_entities, generate_search_queries, synthesize_report |
| ScoringAgent | 全链路评分 + 防幻觉（所有链末端） | evaluate_pipeline_quality ‖ check_hallucination, generate_score_report |

（‖ = ThreadPoolExecutor 并行调用）

## 6. 工具清单（37 个 / 11 模块 / 5 分类，tools/）

| 分类 | 数量 | 工具 |
|------|------|------|
| analysis | 13 | analyze_news_deep, analyze_topic_deep, analyze_kline, generate_kline_analysis, assess_event_impact, synthesize_report, calculate_growth_rate, calculate_financial_ratio, compare_metrics, summarize_financials, evaluate_pipeline_quality, check_hallucination, generate_score_report |
| data | 9 | fetch_stock_news, fetch_financial_news, fetch_announcements, fetch_news_report, fetch_date_events, fetch_kline_context, fetch_kline_report, parse_pdf_file, describe_image_file |
| kb | 7 | kb_load, kb_save, kb_add, kb_remove, kb_stats, chunk_document, chunk_documents_batch |
| extraction | 5 | extract_document_metadata, detect_document_type, extract_financial_metrics, extract_entities, generate_search_queries |
| retrieval | 3 | search_financial_data, query_knowledge_graph, get_graph_stats |

模块文件：core / analysis_tools / news_tools / kline_tools / event_impact_tools /
kb_tools / chunk_tools / extraction_tools / document_parse_tools / report_tools /
scoring_tools / graph_tools（LLM 依赖经 inject_* 函数注入）。

## 7. 基础设施

- **retrievers/**（详见模块内 README）：HybridRetriever = BM25(SQLite FTS5 增量) +
  Chroma 向量 + RRF 融合 + qwen3-rerank + 查询扩展(词典 64 同义组/35 概念映射)；
  QueryPlanner(LLM 拆解复杂查询, 预留 source=graph)；TextChunker 按 doc_type 选策略
  (news 不切/长文按段落)；metadata.py 定义 DocMetadata schema + Chroma 白名单；
  lightrag_adapter 图谱检索。
- **llm/**：DashScopeLLM + LLMCaller(重试/缓存/JSON 平衡括号解析) + ModelRouter(4 档模型)。
- **services/**：analysis(新闻/话题分析 fallback 实现)、conversation(会话管理)、persistence(KB 持久化)。
- **数据**（data/knowledge_base/）：kb_docs.json(知识文档)、news_metadata.json(新闻元数据)、
  bm25_index.db(FTS5)、chroma/(向量库)、embedding_cache.json、conversations/、lightrag/。

## 8. 已知问题 & Backlog（业务通路优先，其余降级）

### P0 — 影响通路正确性（新架构跑通前必须处理）
1. **编排器并发竞态**：orchestrator 是共享单例，`set_pipeline()`+`self.context`
   为可变全局状态，并发请求互相覆写链与上下文（orchestrator.py L128/L151 ↔ pipeline.py L360 ↔ analysis_router.py L151）。
2. **防幻觉静默跳过**：ScoringAgent 无 check_text 时返回 `{}`；`can_handle=False`
   时 orchestrator 无痕 continue — 违反"防幻觉禁止静默跳过"。

### P1 — 通路跑通后第一批清理
3. 双重评分：链内 ScoringAgent 与 `_phase_evolve` 各跑一次 Guard（保留单通道）。
4. 假分数：`_phase_evolve` 硬编码 0.9/0.85/0.8，应换真实质量信号（RAGAS 基线已建）。
5. 重试机制失效：BaseAgent.run 吞掉所有异常，orchestrator 的 max_retries 永不触发。
6. 三条 AnalysisAgent 入口（链 / _run_analysis_via_agent_chain / api_kline 直连）收敛为一条。

### P2 — 历史债（明确降级，不阻塞业务）
7. 死代码：ExecutionMode.PARALLEL/CONDITIONAL、MessageBus(protocol.py)、
   pipeline data_orchestrator 多Pool分支、_route_query fetched_date 死信号。
8. orchestrator.history 无界增长；tool call ID 秒级时间戳会撞。
9. 同步+线程池架构 → async 化；verbose print → logger。
10. IngestionAgent EXPECTED_METADATA_FIELDS 与 DocMetadata schema 脱节（fiscal_period/currency 财报时代残留）。
11. AgentRouter 关键词正则分类 → LLM/Planner 路由；公司名硬编码正则与 STOCK_MAP 双份维护。
12. 测试套件对齐：迭代期暂停把全量 `pytest`（约 594 用例，多写于旧架构）当改动门禁——跑绿 ≠ 通路正确；待新架构定型后按 5 条通路统一重写/精简用例。

## 9. 测试与运行

```cmd
myenv\Scripts\activate.bat
python -m financial_rag.main web        # Web UI: http://127.0.0.1:8000
pytest tests -q                          # 594 用例, 全 mock 数据源, <10s
```

Mock 约定：只 mock 数据源 API（Tushare/RSS），LLM/Embedding/Rerank 始终真实调用。

> 迭代期约定：暂停把全量 `pytest` 当改动门禁（见 §8 P2-12）；需要验证时只对当前正在改的通路做轻量探针（`python -c` 导入/冒烟）。
