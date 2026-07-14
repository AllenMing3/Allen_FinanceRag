---
name: financial-rag-architecture
description: Understand the Financial RAG multi-agent architecture — agent routing (AgentRouter), agent chain scheduling, 5-phase pipeline, data flow (news=metadata vs files=knowledge), function-calling tool system, AI-sector domain focus, hybrid retrieval, and KB lifecycle. Use when modifying agents, pipeline, retrieval, routing, tools, or any code that touches data flow between components.
---

# Financial RAG Architecture

## Domain Focus: AI/科技行业

This system is purpose-built for the **AI/technology sector**. All prompts, metrics, entities, few-shot examples, and document types are optimized for AI industry analysis:

- **11 document types**: 年报、季报、公告、政策文件、新闻报道、研究报告、技术报告、产品发布、融资公告、行业分析、其他
- **4 input formats**: JSONL/TXT (text), PDF (PyMuPDF), PNG/JPG/JPEG/WEBP (qwen-vl-plus 多模态)
- **12 core metrics** across 4 categories: Financial (revenue, rd_expense, arr), Compute (gpu_count, training_cluster_size, inference_cost), Model (model_params, context_window, benchmark_score), Commercial (api_calls, customer_count, dau, mau)
- **9 entity dimensions**: companies, persons, ai_models, chips_hardware, tech_terms, financial_figures, event, industries, key_topics

---

## 1. System Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Layer (FastAPI)                       │
│  web.py → services/analysis.py + services/persistence.py        │
├─────────────────────────────────────────────────────────────────┤
│                     Pipeline Scheduler                           │
│  Phase1:Fetch → Phase2:Index → Phase3:Process → Phase4:Output  │
│                                            → Phase5:Evolve     │
├──────────────┬──────────────┬──────────────────┬────────────────┤
│  Agent Chain │  Hybrid      │  Slot Filler     │  ScoreCard +   │
│  (Orchestr.) │  Retriever   │  (Templates)     │  Hallucination │
│              │  BM25+Vec+   │                  │  Guard         │
│  Ingestion   │  Rerank      │                  │                │
│  Extraction  │  + Chunker   │                  │                │
│  Report      │  + MetaFilter│                  │                │
│  KLine       │  + Persist   │                  │                │
│  EventImpact │              │                  │                │
├──────────────┴──────────────┴──────────────────┴────────────────┤
│                   Function Calling Tools                          │
│  FunctionRegistry → ToolExecutor → ToolCallSession               │
│  Categories: retrieval | analysis | compute | data               │
├─────────────────────────────────────────────────────────────────┤
│              LightRAG Knowledge Graph (PDF/Image only)           │
│  Entity-Relation Extraction → GraphML + Vector Store             │
│  Query modes: local / global / hybrid / mix                      │
├─────────────────────────────────────────────────────────────────┤
│                      Data Sources                                │
│  Tushare (K-line) │ 10jqka/Sina/EastMoney (news) │ Files (KB)   │
├─────────────────────────────────────────────────────────────────┤
│                      LLM Layer (DashScope)                       │
│  Chat (qwen-plus) │ Embedding (text-embedding-v3) │ Rerank      │
│  MultiModal (qwen-vl-plus) — 图片多模态理解               │
│  ModelRouter: auto-selects model by task complexity              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Two Data Roles (Critical Constraint)

| Data Source | Role | Indexed into KB? | Purpose |
|-------------|------|-----------------|---------|
| **Imported Files** (AI reports, filings) | Knowledge | YES — after agent analysis via tools | The actual knowledge base content |
| **Imported PDF/Image** | Knowledge + Graph | YES — also indexed into LightRAG graph | Structured entity-relation graph |
| **News** (domestic APIs: 10jqka/Sina/EastMoney) | Metadata | NO | Parsing prior + query context |
| **K-line** (Tushare) | On-demand | NO | Real-time technical analysis |
| **Events** (date-keyed news) | Mapping | NO | Event→impact analysis (V1) |

**News is NOT knowledge.** It serves two supporting roles:
1. **Parsing Prior** — During file import, news metadata is injected into agent LLM prompts to help extraction
2. **Query Context** — At query time, matching news metadata is shown alongside KB results for temporal context

**K-line and Events are on-demand data** — fetched at query time, never stored in KB.

---

## 3. Agent Architecture (4 Agents)

### Core Principle: Lightweight Decision-Makers

Agents MUST delegate ALL heavy work (data fetching, computation, extraction) to registered tools via `self.call_tool()`. An agent's `process()` method should be ≤100 lines of orchestration logic.

```python
# GOOD: Agent delegates to tools
class AnalysisAgent(BaseAgent):
    def process(self, context):
        metrics = self.call_tool("extract_financial_metrics", text=combined_text)
        entities = self.call_tool("extract_entities", text=combined_text)
        return AgentResult(success=True, data={"metrics": metrics, "entities": entities})

# BAD: Agent embeds logic directly
class BadAgent(BaseAgent):
    def process(self, context):
        # DON'T DO THIS — regex extraction inside agent
        revenue = re.findall(r'营业收入[：:]\s*(\d+)', text)
        return AgentResult(...)
```

### Tool Injection Pattern

```python
# In factory.py:
registry = create_financial_registry(retriever=retriever, llm=llm)
executor = ToolExecutor(registry)
for agent in agents:
    agent.bind_tools(registry, executor)  # Injects FunctionRegistry + ToolExecutor

# In agent code:
result = self.call_tool("tool_name", param1=value1, param2=value2)
```

### Agent Chain Details

| Agent | File | can_handle() | Tool Calls | Output |
|-------|------|------------|-----------|--------|
| **CoordinatorAgent** | `agents/coordinator_agent.py` | always (first in chain) | `classify_query_intent`, `select_agent_chain` | intent + agent_chain + routing metadata |
| **IngestionAgent** | `agents/ingestion_agent.py` | always (BaseAgent) | `extract_document_metadata`, `detect_document_type`, `parse_pdf_file`, `describe_image_file` | Cleaned text + metadata (source, company, date, doc_type, parse_type) |
| **AnalysisAgent** | `agents/analysis_agent.py` | always (unified) | `extract_financial_metrics`, `extract_entities`, `analyze_kline`, `assess_event_impact`, `synthesize_report` | Intent-aware: metrics + entities + K-line/event + Markdown report |
| **ScoringAgent** | `agents/scoring_agent.py` | has final_answer or findings | `evaluate_pipeline_quality`, `check_hallucination`, `generate_score_report` | Per-stage scores + hallucination check + grade (A/B/C/D) |

**Agent consolidation**: The original 7 agents were consolidated to 4. `AnalysisAgent` merges the old ExtractionAgent + KLineAgent + EventImpactAgent + ReportAgent. It reads `context.metadata.intent` (set by Coordinator) to select the right tool chain.

### AgentRouter: Query Intent → Agent Chain

The **AgentRouter** (`core/agent_router.py`) dynamically selects which agent chain to execute based on query intent:

```
Query: "茅台最近走势" → AgentRouter → kline → [CoordinatorAgent, AnalysisAgent, ScoringAgent]
Query: "2024-06-01发生了什么" → AgentRouter → event_impact → [CoordinatorAgent, AnalysisAgent, ScoringAgent]
Query: "商汤科技营收多少" → AgentRouter → report → [CoordinatorAgent, IngestionAgent, AnalysisAgent, ScoringAgent]
Query: "最新AI新闻" → AgentRouter → news → [CoordinatorAgent, IngestionAgent, AnalysisAgent, ScoringAgent]
Query: "其他分析" → AgentRouter → general → [CoordinatorAgent, IngestionAgent, AnalysisAgent, ScoringAgent]
```

**Routing flow in Pipeline Phase 3:**
1. `_route_query()` → calls `AgentRouter.route(query)` → returns `RoutingDecision`
2. `orchestrator.set_pipeline(routing.agent_chain)` → dynamically sets execution order
3. Orchestrator runs agents in chain, each `can_handle()` validates eligibility
4. Router metadata (date, stock_code, intent) is injected into `AgentContext.metadata`

**Key design decisions:**
- Pattern-based classification (keywords + regex), no LLM needed for routing
- `CoordinatorAgent` runs first to set intent for AnalysisAgent
- `ScoringAgent` always runs last to evaluate the full pipeline
- Specialist tool chains (K-line/event) are selected inside AnalysisAgent based on intent

### AgentContext: Shared Data Container

```python
@dataclass
class AgentContext:
    raw_input: str = ""                           # User query
    parsed_data: Any = None                       # IngestionAgent output → ExtractionAgent input
    extracted_features: Dict = field(...)          # ExtractionAgent output → ReportAgent input
    intermediate_findings: List[Dict] = field(...) # Per-agent findings accumulator
    final_answer: Optional[str] = None            # ReportAgent sets this
    metadata: Dict[str, Any] = field(...)          # Free-form: retrieved_items, fetched_data, etc.
```

**Data flow between agents** uses `context_updates` in `AgentResult`:
```python
return AgentResult(
    success=True,
    data={"metrics": metrics},
    context_updates={
        "extracted_features": {"metrics": metrics, "entities": entities},
        "final_answer": analysis_text,
    }
)
```
The orchestrator's `_apply_updates()` merges these into the shared AgentContext.

---

## 4. Hybrid Retrieval Engine

### Architecture: 3 Channels + Chunking + Filtering + Persistence

```
Query ──→ BM25 (keyword, bigram tokenizer) ─┐
          Vector (text-embedding-v3, 1024d) ─┼─→ RRF Fusion → qwen3-rerank → Metadata Filter → Top-K
          Jaccard (fallback, no API)       ─┘
```

### Three Retrieval Modes (auto-detected)

| Mode | Condition | Pipeline |
|------|-----------|----------|
| Local only | No API Key | BM25 + Jaccard → RRF |
| With Embedding | API Key set | BM25 + Vector → RRF |
| Full pipeline | API Key + Rerank | BM25 + Vector → RRF → qwen3-rerank |

### BM25 Implementation Details

- **Tokenizer**: Priority: injected jieba → regex fallback with Chinese bigram sliding window
- **Length normalization**: `dl`/`avg_dl` in **token count** (not character count), precomputed in `_doc_token_lens`
- **Inverted index dedup**: `_build_bm25_index()` and `add()` both use `set(tokens)` to prevent duplicate entries
- **Parameters**: k1=1.2, b=0.75 (standard BM25)

### TextChunker (Long Document Splitting)

```python
from financial_rag.retrievers import TextChunker, HybridRetriever

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
retriever = HybridRetriever(chunker=chunker, embedder=..., reranker=...)
retriever.index(docs)  # Auto-chunks long documents before indexing
```

- **Priority splitting**: paragraphs (`\n\n`) → sentences (`。！？`) → hard split
- **Greedy merge**: Small chunks merged into previous (min_chunk_size=50)
- **Overlap**: Adjacent chunks share `chunk_overlap` chars for context continuity
- **Metadata preservation**: Each chunk inherits parent meta + adds `chunk_id`, `source_id`, `chunk_start`

### Metadata Filtering

```python
# Exact match
results = retriever.search("AI芯片", filters={"source": "news"})

# Multi-value (OR)
results = retriever.search("芯片", filters={"source": ["news", "analysis"]})

# Range filter
results = retriever.search("茅台", filters={"publish_time": {"gte": "2024-06-01"}})

# Combined (AND between keys)
results = retriever.search("AI", filters={"source": "news", "date": {"gte": "2024-01-01"}})
```

Filters apply AFTER RRF fusion and rerank, narrowing the candidate set.

### Index Persistence

```python
# Save entire index state (docs + BM25 + embeddings)
retriever.save_index("output/kb_index.json")

# Load without re-embedding (saves API cost + time)
retriever2 = HybridRetriever()
retriever2.load_index("output/kb_index.json")
# → Instant restore: 100 docs, BM25 index, embedding vectors all restored
```

Saved format: JSON with `version`, `documents`, `doc_embeddings`, `bm25_index`, `doc_token_lens`, `config`.

---

## 5. 5-Phase Pipeline

```python
scheduler = PipelineScheduler(orchestrator, retriever, registry, executor, llm, filler)
result = scheduler.run("茅台2024年营收增长了多少？")
```

| Phase | Method | Input | Output | Key Behavior |
|-------|--------|-------|--------|-------------|
| **1: Fetch** | `_phase_fetch()` | query | `fetched_data`, `tool_call_stats` | FC session auto-selects news/K-line tools. Max 2 rounds. |
| **2: Index** | `_phase_index()` | `fetched_data` | `indexed_docs`, `retrieved_items` | **Uses `add()` (not `index()`) if KB already has docs** — preserves existing KB. |
| **3: Process** | `_phase_process()` | `retrieved_items` | `agent_exec_result` | Builds `AgentContext` with `retrieved_items` + `fetched_data` in metadata. Runs orchestrator. |
| **4: Output** | `_phase_output()` | agent results + retrieved + fetched | `final_output`, `fill_stats` | **Agent findings are prepended** to context_docs (Phase 3 output flows to Phase 4). |
| **5: Evolve** | `_phase_evolve()` | all previous results | `scorecard`, `hallucination_check` | PipelineScoreCard per-stage scoring + HallucinationGuard verification. |

### Pipeline Data Flow (Critical)

```
Phase 1 (fetch) ─── fetched_data ──────────────────────────┐
Phase 2 (index) ─── retrieved_items ──────────────────┐    │
Phase 3 (process) ─ agent_exec_result ──────────┐     │    │
                                                 ↓     ↓    ↓
Phase 4 (output):  context_docs = [agent_analysis] + [retrieved] + [fetched]
                                                 ↓
Phase 5 (evolve):  ScoreCard + HallucinationGuard checks final_output against retrieved_items
```

**Key pitfall**: Phase 2 must NOT call `retriever.index()` on fetched data if KB already has documents — it would wipe the existing KB. Use `retriever.add()` for incremental indexing.

---

## 6. Tool System (Function Calling)

### Registry Architecture

```
FunctionRegistry (central hub)
├── [retrieval]  search_financial_data — KB search (retriever injected)
├── [analysis]   calculate_growth_rate, calculate_financial_ratio, compare_metrics, summarize_financials
├── [compute]    (via extraction_tools) extract_financial_metrics, extract_entities, etc.
├── [data]       fetch_stock_news, fetch_financial_news, fetch_announcements
│                fetch_etf_kline_report, fetch_news_report
│                fetch_date_events, fetch_kline_context
│                describe_image_file, parse_pdf_file
│                query_knowledge_graph, get_graph_stats
└── [analysis]   assess_event_impact (LLM or keyword fallback)
```

### Tool Categories

| Category | Purpose | Examples |
|----------|---------|---------|
| `retrieval` | Search indexed KB | `search_financial_data` |
| `analysis` | Compute ratios, compare, assess | `calculate_growth_rate`, `assess_event_impact` |
| `compute` | Generic math/stats | (extensible) |
| `data` | Fetch real-time data | `fetch_stock_news`, `fetch_etf_kline_report`, `fetch_date_events` |

### ToolCallSession: Multi-Round FC

```python
session = ToolCallSession(llm=llm, registry=registry, max_rounds=5)
stats = session.run("分析茅台最近走势")
# Round 1: LLM calls fetch_etf_kline_report
# Round 2: LLM gets results, generates final answer
# stats.calls = [ToolCallResult(name="fetch_kline_report", success=True, ...)]
```

### Extraction Tools Strategy

All extraction tools use **LLM-first + regex-fallback**:
1. Try LLM structured output (JSON parsing)
2. On failure: fall back to regex pattern matching
3. Every result includes `_confidence` (high/medium/low/none) and `_source` (llm/regex)

### Document Parse Tools (非文本文档解析)

`tools/document_parse_tools.py` 提供两类非文本文件的解析能力，遵循与 extraction_tools 相同的闭包注入 + FunctionDef 注册模式：

| Tool | 输入 | 策略 | 输出 |
|------|------|------|------|
| `describe_image_file` | png/jpg/jpeg/webp | **LLM-first** — qwen-vl-plus 多模态理解，domain prompt 在 `prompts.py` (IMAGE_UNDERSTANDING_SYSTEM/PROMPT + few-shot) | description + vision_model + _confidence |
| `parse_pdf_file` | .pdf | **纯本地** — PyMuPDF 逐页提取文本 | text + page_count + char_count + _confidence |

**数据流**: IngestionAgent `_ingest_file()` 按扩展名分发 → `_ingest_pdf()` / `_ingest_image()` → `self.call_tool("parse_pdf_file"/"describe_image_file")` → 解析结果走标准 `_ingest_text()` 流程（清洗 → 相关性门控 → 分类 → 元数据抽取）。解析结果补充 `parse_type: "pdf"/"image"` 元数据。

**API 层**: `ingest_router.py` 的 `_parse_pdf_text()` / `_describe_image()` 直接复用 `document_parse_tools` 的函数，不重复实现。

---

## 7. Service Layer (web.py → services/)

web.py is a **thin HTTP layer**. Business logic lives in `services/`:

| Service | File | Purpose |
|---------|------|----------|
| `analysis` | `services/analysis.py` | `analyze_news_text()`, `analyze_topic_research()` — pure functions |
| `persistence` | `services/persistence.py` | `load_kb()`, `save_kb()`, `load_meta()`, `save_meta()`, `assign_doc_ids()`, `dedup_docs()`, `make_doc_id()` — disk I/O + KB resilience |

**Pattern**: Services take dependencies as parameters (llm, retriever), never import FastAPI. This enables unit testing without HTTP.

---

## 8. KB Lifecycle (Hardened)

```
Server Start → persistence.load_kb() → kb_docs buffer (in-memory)
  ↓
Assign doc_ids (idempotent, SHA-256 hash of source+text[:200])
  ↓
Check persisted index (kb_index.json):
  ├── Index exists & doc count matches → load_index() → kb_built=True (NO re-embed!)
  └── Index missing/stale → retriever.index(kb_docs) → save_index(kb_index.json)
  ↓
Ingest Files → assign_doc_ids → dedup_docs (skip existing) → retriever.add() (incremental)
  ↓
Subsequent Queries → Hybrid retrieval against indexed KB (no re-index)
  ↓
Phase 1 Fetch → New news/K-line data → retriever.add(fetched_docs) — incremental, preserves KB
  ↓
Delete/Clear → Full rebuild (retriever.index) → save_index() (or delete index file if empty)
  ↓
Shutdown → persistence.save_kb() + save_meta() + retriever.save_index()
```

**KB resilience features:**
- **Index persistence**: `kb_index.json` stores docs + embeddings + BM25 index. Loaded on startup to skip expensive re-embedding.
- **Doc identity**: Every doc gets a stable `doc_id` (SHA-256 of source+text[:200]). Used for deduplication.
- **Ingest dedup**: `dedup_docs()` filters out docs whose doc_id already exists in KB.
- **Auto-backup**: `_atomic_write_json()` creates `.bak` copy before each write.
- **News archive rotation**: `append_news_archive()` auto-trims to 5000 lines max.

**Auto-build pattern** (used in web.py endpoints):
```python
r = _state["retriever"]
if not _state.get("kb_built"):
    docs = _state.get("kb_docs", [])
    if not docs:
        raise HTTPException(400, "知识库为空，请先导入数据")
    r.clear()
    r.index(docs, precompute_embeddings=True)
    r.save_index(_INDEX_PATH)  # persist for next startup
    _state["kb_built"] = True
```

---

## 9. Scoring & Anti-Hallucination

### PipelineScoreCard (`core/scorer.py`)

Per-stage scoring with independent metrics:
- Tokenization → BM25 → Vector → RRF → Rerank → Chunk → LLM → Hallucination
- Each stage: score (0-1.0) + diagnosis + elapsed_ms
- Grade: A (≥0.90), B (≥0.75), C (≥0.50), D (<0.50)

### HallucinationGuard (`guard/reflector.py`)

6-layer progressive verification:
1. Source coverage — are claims backed by retrieved items?
2. Numerical consistency — do numbers match sources?
3. Temporal coherence — are dates/times consistent?
4. Entity consistency — same entities across sources?
5. Logical coherence — does the answer follow from sources?
6. Confidence calibration — is the answer appropriately hedged?

### ReflectionLoop (`guard/reflector.py`)

ReAct cycle: Think → Retrieve → Act → Observe → Judge
- Max `max_retrievals` retrieval rounds
- Multi-dimensional confidence scoring
- Auto-stops when `min_confidence` reached

---

## 10. Mock Mode Rules

**Scope**: Mock ONLY data source APIs. LLM/Embedding/Rerank are ALWAYS real.

| Component | Mocked? | How |
|-----------|---------|-----|
| Tushare K-line | YES | `MOCK_MODE=true` → synthetic OHLCV data |
| News APIs (10jqka/Sina/EastMoney) | YES | `MOCK_MODE=true` → pre-built news fixtures |
| LLM (qwen-plus) | NO | Always real API call |
| Embedding (text-embedding-v3) | NO | Always real API call |
| Rerank (qwen3-rerank) | NO | Always real API call |

---

## 11. Configuration (`config.py`)

| Section | Key Params | Defaults |
|---------|-----------|----------|
| `config.llm` | model, embedding_model, rerank_model, temperature | qwen-plus, text-embedding-v3, qwen3-rerank, 0.0 |
| `config.rag` | chunk_size, chunk_overlap, similarity_top_k | 512, 50, 5 |
| `config.coordinator` | execution_mode, max_parallel_agents, max_retries | sequential, 3, 2 |
| `config.pipeline` | hybrid_top_k, rrf_k, bm25_weight, vector_weight | 10, 60, 0.3, 0.7 |
| `config.reflection` | max_retrievals, max_steps, min_confidence | 3, 6, 0.6 |
| `config.tushare` | token, default_stocks | env, [600519, 000858, ...] |
| `config.mock` | enable | env MOCK_MODE |

---

## 12. Key Files Reference

| Layer | Files |
|-------|-------|
| **Agents** | `agents/coordinator_agent.py`, `agents/ingestion_agent.py`, `agents/analysis_agent.py` (unified), `agents/scoring_agent.py`, `agents/utils.py` |
| **Core** | `core/base.py` (BaseAgent+AgentContext+AgentResult), `core/orchestrator.py`, `core/pipeline.py`, `core/indexer.py`, `core/factory.py`, `core/agent_router.py`, `core/protocol.py`, `core/router.py`, `core/scorer.py`, `core/data_orchestrator.py` |
| **Retrieval** | `retrievers/retriever.py` (HybridRetriever), `retrievers/bm25_engine.py`, `retrievers/vector_engine.py`, `retrievers/fusion.py`, `retrievers/filters.py`, `retrievers/chunker.py`, `retrievers/persistence.py`, `retrievers/preprocessor.py`, `retrievers/query_parser.py`, `retrievers/dictionaries.py` |
| **Tools** | `tools/core.py` (FunctionRegistry+ToolExecutor+ToolCallSession), `tools/extraction_tools.py`, `tools/document_parse_tools.py`, `tools/graph_tools.py`, `tools/news_tools.py`, `tools/kline_tools.py`, `tools/event_impact_tools.py`, `tools/coordinator_tools.py`, `tools/scoring_tools.py`, `tools/report_tools.py` |
| **Guard** | `guard/reflector.py` (HallucinationGuard + ReflectionLoop) |
| **Services** | `services/analysis.py` (pure business logic), `services/persistence.py` (KB/metadata/index I/O + dedup + backup) |
| **LLM** | `llm/caller.py` (LLMCaller smart layer), `llm/dashscope_client.py`, `llm/model_router.py` |
| **Graph** | `retrievers/lightrag_adapter.py` (LightRAG sync wrapper), `tools/graph_tools.py` (FC graph query) |
| **Data** | `rss_fetcher.py` (news: 10jqka/Sina/EastMoney), `tushare_client.py` (K-line + financials) |
| **Quality** | `core/scorer.py` (PipelineScoreCard), `guard/reflector.py` (HallucinationGuard) |
| **Web** | `web.py` (FastAPI endpoints), `static/` (HTML/CSS/JS frontend) |
| **Config** | `config.py` (all settings), `.env` (API keys) |

---

## 13. Common Pitfalls (Agent Code Quality)

1. **Never embed extraction/compute logic in agents** — always delegate to tools. If you find yourself writing regex or math in `process()`, create a tool instead.

2. **Never call `retriever.index()` on incremental data** — use `retriever.add()` to preserve existing KB documents.

3. **Always call `save_index()` after any full rebuild** — otherwise next startup will re-embed everything.

4. **Always `assign_doc_ids()` before storing docs** — without doc_ids, dedup won't work and re-ingesting the same file will double the KB.

5. **Agent `context_updates` must use correct field names** — `final_answer`, `intermediate_findings`, `extracted_features`, `parsed_data`, `metadata`. The orchestrator's `_apply_updates()` only sets fields that exist on `AgentContext`.

6. **Pipeline data flow must be unidirectional**: Phase 1→2→3→4→5. Each phase reads from `PipelineResult` fields set by previous phases, never reaches back.

7. **Tool callbacks must be pure functions** — no side effects on global state. Dependencies (retriever, llm) are injected via closures or `inject_*()` functions.

8. **BM25 dl/avg_dl must be in token units** — precomputed in `_doc_token_lens`, not character count.

9. **Metadata filter applies AFTER RRF+Rerank** — it narrows candidates, doesn't replace retrieval.

10. **Guard module is at `guard/reflector.py`**, NOT `core/reflector.py`.
