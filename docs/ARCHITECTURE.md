# Architecture Deep Dive

> This document provides detailed architecture documentation. See [README.md](../README.md) for project overview and quick start.

## Component Responsibilities

| Engine | File | Responsibility |
|--------|------|----------------|
| **Route** | `core/agent_router.py` | Intent classification (5 domains), agent chain selection, metadata extraction (date/stock) |
| **Coordinate** | `core/orchestrator.py` | Register agents, decide execution order, pass context. Metadata merge (not replace), list extend |
| **Data Orchestrate** | `core/data_orchestrator.py` | Multi-pool text management: TextPreprocessor → DocTypeClassifier → KnowledgePool routing, cross-pool search |
| **Schedule** | `core/pipeline.py` | 5-phase pipeline: Fetch → Index → Process (AgentRouter) → Output (SlotFiller, skippable) → Evolve (Scoring + HallucinationGuard) |
| **Indexer** | `core/indexer.py` | 4-stage retrieval: Clean → Extract → Retrieve → Verify. BM25 + ChromaDB + RRF fusion |
| **Reflect** | `guard/reflector.py` | ReAct loop (Think → Act → Observe → Judge) + 4-layer anti-hallucination guard (source grounding, numerical fidelity, citation integrity, structure compliance) |
| **Score** | `core/scorer.py` | Full-pipeline scorecard: phase coverage, hallucination, citation density, answer relevance |

---

## Knowledge Base Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│                       Data Sources                           │
│  News (10jqka / Sina / EastMoney) · File Import · Tushare    │
└──────────┬──────────────────┬──────────────────┬─────────────┘
           │                  │                  │
     ┌─────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
     │  News     │    │  File Import│    │  KLine      │
     │  metadata │    │  Agent Chain│    │  On-demand  │
     │  archive  │    │  IA → EA    │    │  report     │
     └─────┬─────┘    └──────┬──────┘    └─────────────┘
           │                 │
  ┌────────▼────────┐  ┌─────▼──────┐
  │ news_metadata   │  │  kb_docs   │
  │ news_archive    │  │  .json     │
  └────────┬────────┘  └──┬─────┬───┘
           │              │     │
     context         ┌────▼──┐ ┌▼──────┐
     injection       │ BM25  │ │Chroma │
                     │ Index │ │DB ANN │
                     └───┬───┘ └──┬────┘
                         │        │
                    ┌────▼────────▼────┐
                    │  HybridRetriever │
                    │  RRF + Rerank    │
                    │  TextChunker     │
                    │  Metadata Filter │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  AgentRouter     │
                    │  Intent → Chain  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Answer + Score  │
                    └──────────────────┘
```

| File | Purpose |
|------|--------|
| `data/knowledge_base/kb_docs.json` | Analyzed KB documents — loaded on server start, saved after file import with agent analysis |
| `data/knowledge_base/news_metadata.json` | News context labels — used as **parsing prior** and **query-time context** |
| `data/knowledge_base/news_archive.jsonl` | Cumulative raw news archive — each search appends with full metadata |
| `output/*.md` | Markdown reports (news summaries, K-line analysis reports) |

### KB Management APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kb/status` | GET | KB doc count + source breakdown |
| `/api/kb/sources` | GET | List all sources with doc counts |
| `/api/kb/search` | GET | Search KB docs by keyword (`?keyword=xxx`) |
| `/api/kb/keyword/{kw}` | DELETE | Delete all docs matching a keyword |
| `/api/kb/source/{name}` | DELETE | Delete all docs from a source |
| `/api/kb/clear` | POST | Wipe all docs + reset ingestion progress |
| `/api/kb/history` | GET | List analysis conclusions (learning history) |
| `/api/ingest/progress` | GET | Poll background ingestion progress |
| `/api/file/preview` | GET | Preview first N lines of a file (`?path=xxx&file=yyy&lines=20`) |
| `/api/ingest/files` | POST | Import selected files with analysis mode (`files: [...]`, `mode: "deep"|"quick"`) |

---

## Agent Routing

`AgentRouter` is the query-time decision engine. It runs in Phase 3 (Process) and determines which agent chain executes.

### Intent Classification

Keyword and pattern matching with confidence scoring:

| Intent | Trigger Patterns | Confidence Threshold |
|--------|-----------------|---------------------|
| `kline` | K-line terms (MACD, RSI, 支撑位), stock codes (SH/SZ), stock names from `STOCK_MAP` | 0.5 |
| `event_impact` | Event keywords (并购, 重组, 利好/利空, 涨停), date patterns (YYYY-MM-DD, YYYY年M月) | 0.5 |
| `report` | Financial terms (营收, 净利润, 财报, 年报/季报, EPS, ROE) | 0.5 |
| `news` | News terms (新闻, 动态, 最新消息, 行业) | 0.5 |
| `general` | Fallback — all other queries | — |

### Chain Selection

Every chain ends with `ScoringAgent` for quality assurance. `CoordinatorAgent` is **never** in chains (pipeline handles routing).

| Intent | Chain |
|--------|-------|
| `kline` | AnalysisAgent (intent=kline) → ScoringAgent |
| `event_impact` | AnalysisAgent (intent=event_impact) → ScoringAgent |
| `report` | IngestionAgent → AnalysisAgent (intent=general) → ScoringAgent |
| `news` | IngestionAgent → AnalysisAgent (intent=news, deep analysis) → ScoringAgent |
| `general` | IngestionAgent → AnalysisAgent (intent=general) → ScoringAgent |

**Low-confidence override:** When intent confidence < 0.5, `IngestionAgent` is prepended to ensure context gathering before downstream agents.

### Metadata Extraction

Alongside intent classification, `AgentRouter` extracts structured metadata:

- **Date**: `YYYY-MM-DD`, `YYYY年M月D日`, `YYYYMMDD` → stored in `routing_decision.metadata.date`
- **Stock**: keyword lookup via `STOCK_MAP` (11 stocks), SH/SZ code patterns → stored in `routing_decision.metadata.ts_code`
- **Context override**: If query already contains `ts_code` in orchestrator context, it is preserved

---

## Agent Chain (Phase 3 Detail)

Agents are **lightweight orchestrators** — all business logic delegated to registered tools via Function Calling. `AnalysisAgent` consolidates 5 former agents (Extraction, KLine, EventImpact, Report) and selects its tool chain based on `context.metadata["intent"]`.

### Report / General Chain

```
IngestionAgent
  → call_tool(extract_document_metadata)
  → call_tool(detect_document_type)
  → context →

AnalysisAgent (intent=general)
  → call_tool(extract_financial_metrics)
  → call_tool(extract_entities)
  → call_tool(generate_search_queries)
  → call_tool(synthesize_report)
  → context →

ScoringAgent
  → call_tool(evaluate_pipeline_quality)
  → call_tool(check_hallucination)
  → call_tool(generate_score_report)
```

### News Deep Analysis Chain

```
IngestionAgent
  → call_tool(extract_document_metadata)
  → context.parsed_data →

AnalysisAgent (intent=news, has parsed_data)
  → call_tool(analyze_news_deep)       # wraps services/analysis.py
  → _render_structured_news()          # multi-dim impact, key signals, risks
  → context →

ScoringAgent
```

### K-Line Chain

```
AnalysisAgent (intent=kline)
  → call_tool(fetch_kline_report)     # STOCK_MAP resolve + Tushare fetch
  → call_tool(analyze_kline)          # MACD / RSI / Bollinger / KDJ
  → call_tool(generate_kline_analysis) # LLM interpretation
  → context →

ScoringAgent
```

### Event Impact Chain

```
AnalysisAgent (intent=event_impact)
  → call_tool(fetch_date_events)       # Date-based event retrieval
  → call_tool(assess_event_impact)     # Bullish / bearish + impact factor
  → context →

ScoringAgent
```

### AI Industry Metrics (AnalysisAgent, intent=general)

| Category | Metrics |
|----------|---------|
| Financial | revenue, net_income, gross_margin, rd_expense, arr |
| Compute | gpu_count, training_cluster_size, inference_cost_per_token, compute_utilization |
| Model | model_params, context_window, inference_latency, benchmark_score |
| Commercial | api_calls, customer_count, dau, mau |

**Entity extraction dimensions:** companies, persons, ai_models, chips_hardware, tech_terms, financial_figures, event, industries

---

## LLM Call Layer

All LLM calls in tools are wrapped by `LLMCaller` for robustness — no bare `llm.chat()` calls remain.

| Capability | Description |
|------------|-------------|
| **Retry** | Exponential backoff (0.5s → 1s → 2s) on transient errors |
| **Structured JSON** | `call_json()` auto-retries on parse failure, balanced bracket parsing (handles nested strings/escapes), appends JSON output hint |
| **Caching** | Hash-based response cache with configurable TTL |
| **Input Validation** | Max-length check before sending to API |
| **Anti-Hallucination** | Default system constraints: no fabricated data, no speculation |

SlotFiller also uses `LLMCaller` for retry + caching + input validation. Pipeline Phase 4 (Output/SlotFiller) is **skipped** when Phase 3 (Agent) already produced >50 chars of output — no redundant template filling.

```python
from financial_rag.llm import LLMCaller, get_caller

# Direct wrapping
caller = LLMCaller(llm)
result = caller.call_json("Extract metrics from this report", system="...")

# Via ModelRouter
caller = router.get_caller_for_agent("analysis")
text = caller.call("Generate summary", temperature=0.3)
```

---

## Orchestrator Metadata Merge

`orchestrator._apply_updates()` uses **merge semantics** — not wholesale replacement — to prevent downstream agents from wiping upstream data:

| Update Type | Behavior | Example |
|-------------|----------|---------|
| **Dict attribute** (e.g. `metadata`) | `current.update(v)` — merge, not replace | AnalysisAgent adds `ts_code`, ScoringAgent adds `report_type` → both preserved |
| **List attribute** (e.g. `intermediate_findings`) | `current.extend(v)` — append, not replace | Each agent's findings accumulate across the chain |
| **Scalar attribute** (e.g. `final_answer`) | `setattr(self.context, k, v)` — direct replace | Last agent's answer wins |
| **Unknown key** | Written to `context.metadata` | Custom fields from any agent land in metadata |

---

## Retrieval Modes

| Mode | Condition | Chain |
|------|-----------|-------|
| Local only | No API Key | BM25 + Jaccard (in VectorEngine) → RRF |
| With Embedding | API Key set | BM25 + ChromaDB ANN (1024-dim) → RRF |
| Full pipeline | API Key active | BM25 + ChromaDB ANN → RRF → qwen3-rerank |

`HybridRetriever` also applies `TextChunker` (split + overlap + metadata tagging) at index time and metadata filtering at query time. ChromaDB `PersistentClient` stores vectors on disk (`data/knowledge_base/chroma/`); content-hash MD5 document IDs ensure stable identity across add/remove operations.

**Query Expansion** (规则优先 + LLM 增强):
- **同义词扩展** (weight=1.5): 35 组双向同义词，如 "英伟达" ↔ "NVIDIA" ↔ "NVDA"，O(1) 查找表
- **概念关联** (weight=0.6): 18 个行业概念单向关联，如 "芯片" → [半导体, 光刻, 晶圆, 封装, 制程]
- **LLM 增强**: 短查询 (<15 字) 且规则扩展词 <2 个时，通过 `llm_rewrite_query()` 补充 2-3 个语义关键词
- BM25 用扩展后加权关键词搜索，ChromaDB 用 `expanded_query` (原 query + 扩展词拼接) 做语义检索

**Efficient deletion:** `HybridRetriever.remove(indices)` filters out docs + syncs ChromaDB deletion by content-hash ID, rebuilds only BM25 (cheap) — avoiding a full `clear() + index()` cycle when deleting by keyword or source.

## Performance Optimizations

| Optimization | Description | File |
|-------------|-------------|------|
| **Background startup** | `_ensure_init()` runs in a background thread via FastAPI `lifespan` — server accepts requests immediately; endpoints still call `_ensure_init()` (idempotent, blocks if not ready) | `web.py` |
| **Async endpoints** | All endpoints are `async def` with blocking calls wrapped in `asyncio.to_thread()` — FastAPI handles concurrent requests without serialization | `api/*.py` |
| **Parallel extraction** | `AnalysisAgent._run_extraction_chain()` runs `extract_financial_metrics` + `extract_entities` in parallel via `ThreadPoolExecutor(2)` | `agents/analysis_agent.py` |
| **Phase 4 skip** | Pipeline skips SlotFiller output when Phase 3 Agent already produced >50 chars — avoids overwriting good agent output with template filler | `core/pipeline.py` |
| **Text cleaning pipeline** | Pre-indexing pipeline: `TextPreprocessor` (boilerplate removal + paragraph dedup enabled by default) → trigram BM25 tokenization → ChromaDB indexing. MD5-based doc IDs for stable RRF dedup | `retrievers/bm25_engine.py`, `retrievers/preprocessor.py`, `retrievers/fusion.py` |
| **SlotFiller LLMCaller** | SlotFiller wraps all LLM calls via `LLMCaller` (retry + cache + input validation) instead of bare `llm.chat()`. TTFT measured from actual latency, not fixed formula | `slot_filler.py`, `llm/caller.py` |
| **Efficient KB deletion** | `HybridRetriever.remove(indices)` filters docs + syncs ChromaDB deletion, rebuilds only BM25 — no full re-embedding | `retrievers/retriever.py` |
| **Compiled regex patterns** | Module-level `re.compile()` for keyword scanning (sentiment, doc-type, event impact) — replaces per-call `kw in text` loops | `services/analysis.py`, `tools/extraction_tools.py`, `tools/event_impact_tools.py` |
| **Precomputed lookup sets** | `frozenset` keyword collections + `_ALL_METRIC_ALIASES` set for O(1) alias exclusion — replaces nested loop scans | `tools/extraction_tools.py` |
| **ChromaDB lazy init** | `PersistentClient` only created when first vector indexed; in-memory fallback when no persist dir | `retrievers/vector_engine.py` |
| **Config TTL cache** | `/api/config` caches result for 60s, avoiding repeated config reads | `api/analysis_router.py` |
| **Query expansion** | 35 组同义词 + 18 个概念关联，规则层零延迟扩展 + LLM 短查询增强。BM25 用加权扩展词，ChromaDB 用 expanded_query | `retrievers/query_parser.py`, `retrievers/dictionaries.py` |
| **HTML cache** | `index.html` read once at startup, served from memory | `web.py` |
| **Orchestrator retry** | `max_retries=1`, retry delay `0.1s` (was 2 retries, 1s delay) | `core/factory.py`, `core/orchestrator.py` |
| **Fetch normalization** | Pipeline `_phase_fetch` normalizes all tool results to standard doc format (`title`, `content`, `source`, `publish_time`, `url`) — handles alternate keys (`items` / `results`) | `core/pipeline.py` |

---

## Model Routing

| Tier | Model | Use case |
|------|-------|----------|
| LIGHT | qwen-turbo | Simple parsing, formatting |
| STANDARD | qwen-plus | Extraction, classification |
| HEAVY | qwen-max | Multi-step reasoning, trend analysis |
| ULTRA | qwen3-235b | Comprehensive analysis, report generation |

---

## File-by-File Guide

### Root

| File | Role |
|------|------|
| `main.py` | Thin wrapper — delegates to `financial_rag.main.main()` |
| `.env.example` | Env var template (`DASHSCOPE_API_KEY`, `TUSHARE_TOKEN`, `MOCK_MODE`) |
| `requirements.txt` | pip dependencies (incl. fastapi, uvicorn) |

### `financial_rag/` — Core Package

| File | Role | Key exports |
|------|------|-------------|
| `__init__.py` | Package entry, version `2.0.0` | Re-exports all public APIs |
| `main.py` | CLI entry (argparse) | `main()` |
| `config.py` | Global config dataclasses | `config`, `AppConfig`, `LLMConfig` |
| `prompts.py` | AI-sector LLM prompt templates + few-shot examples (SenseTime / NVIDIA / ZhipuAI) | — |
| `templates.py` | 4 slot templates: QUICK_QA, FINANCIAL_REPORT, NEWS_BRIEF, DEEP_ANALYSIS | `SlottedTemplate`, `ALL_TEMPLATES` |
| `slot_filler.py` | Parallel slot filling engine with LLMCaller wrapper (retry + cache) + measured TTFT | `SlotFiller`, `create_slot_filler` |
| `rss_fetcher.py` | Financial news via domestic APIs (10jqka / Sina / EastMoney) + rate limiting | `search_news`, `fetch_all_news` |
| `tushare_client.py` | K-line & financial indicators via Tushare Pro | `fetch_stock_kline`, `compute_technical_indicators` |
| `mock_data.py` | 25 AI-sector mock news + 3 long-form articles | Mock data for offline dev |
| `web.py` | FastAPI app wiring + router registration (90 lines) — thin shell that imports 4 routers from `api/` | FastAPI app, lifespan background init, signal-based shutdown |

### `financial_rag/api/` — FastAPI Modular Routers

| File | Role |
|------|------|
| `__init__.py` | Package marker |
| `app_state.py` | Shared singleton state (`_state` dict), lazy `_ensure_init()` with double-check locking, `_persist_state()` |
| `models.py` | Pydantic request models (QueryRequest, NewsRequest, KlineRequest, etc.) |
| `kb_router.py` | KB management: status, query, build, clear, delete by source/keyword, learning history/stats |
| `ingest_router.py` | File preview, directory listing, file/news ingestion with background thread analysis |
| `analysis_router.py` | Config (TTL cached), news/topic analysis, metadata clear/status, news, kline endpoints |
| `query_router.py` | Pipeline, slot fill, scoring endpoints |

All endpoints are `async def` — blocking calls wrapped in `asyncio.to_thread()` for non-blocking event loop.

### `financial_rag/core/` — Architecture Layer

| File | Role | Key exports |
|------|------|-------------|
| `base.py` | Abstract foundations | `BaseAgent`, `AgentContext`, `AgentResult`, `ExecutionMode` |
| `agent_router.py` | Query-time routing: intent classification, chain selection, metadata extraction | `AgentRouter`, `RoutingDecision` |
| `orchestrator.py` | Multi-agent scheduling engine — dict merge, list extend, scalar replace | `AgentOrchestrator` |
| `data_orchestrator.py` | Multi-pool text management: TextPreprocessor + DocTypeClassifier + KnowledgePool routing | `DataOrchestrator`, `KnowledgePool` |
| `pipeline.py` | 5-phase PipelineScheduler (Fetch → Index → Process (AgentRouter) → Output (SlotFiller, skippable) → Evolve (Scoring + HallucinationGuard)) | `PipelineScheduler`, `PipelineResult` |
| `router.py` | CLI command dispatch + handlers | `CommandRouter` |
| `factory.py` | Factory: creates and wires 4 agents + AgentRouter | `create_orchestrator`, `setup_environment` |
| `indexer.py` | Hybrid retrieval pipeline orchestration | `PipelineOrchestrator` |
| `scorer.py` | Full-pipeline scorecard | `PipelineScoreCard`, `ScoreGrade` |
| `protocol.py` | Agent messaging infrastructure | `AgentMessage`, `MessageBus` |

### `financial_rag/agents/` — 4 Agents

| File | Role |
|------|------|
| `coordinator_agent.py` | Intent classification + chain selection via `call_tool(classify_query_intent, select_agent_chain)` |
| `ingestion_agent.py` | Data ingestion → `call_tool(extract_document_metadata, detect_document_type)` |
| `analysis_agent.py` | Unified analysis: routes by `context.metadata["intent"]` — extraction, K-line, event impact, deep news analysis |
| `scoring_agent.py` | Quality scoring → `call_tool(evaluate_pipeline_quality, check_hallucination, generate_score_report)` |
| `utils.py` | Shared: `build_news_context()` |

### `financial_rag/tools/` — 27 Registered Tools across 9 Modules

| File | Tools | Role |
|------|-------|------|
| `core.py` | — | Infrastructure: `FunctionDef`, `FunctionRegistry`, `ToolExecutor`, `ToolCallSession` |
| `extraction_tools.py` | 5 | Document metadata, doc type, financial metrics, entities, search queries (LLM-first + regex fallback) |
| `news_tools.py` | 4 | Fetch stock news, financial news, announcements, news report (10jqka / Sina / EastMoney) |
| `kline_tools.py` | 4 | Fetch K-line report, K-line context, analyze K-line, generate K-line analysis. Also hosts `STOCK_MAP`, `KLINE_ANALYSIS_SYSTEM`, `KLINE_ANALYSIS_PROMPT` |
| `event_impact_tools.py` | 2 | Fetch date events, assess event impact (bullish / bearish + impact factor) |
| `scoring_tools.py` | 3 | Evaluate pipeline quality, check hallucination, generate score report |
| `coordinator_tools.py` | 2 | Classify query intent, select agent chain |
| `report_tools.py` | 1 | Synthesize report (LLM-driven or heuristic fallback) |
| `analysis_tools.py` | 2 | Deep analysis: `analyze_news_deep` (wraps services/analysis.py for multi-dim impact), `analyze_topic_deep` (sub-topics, key players, sentiment trend) |
| `__init__.py` | — | `create_financial_registry()` — registers all 27 tools; re-exports `STOCK_MAP` |

### `financial_rag/retrievers/` — Modular Retrieval Stack

| File | Role |
|------|------|
| `retriever.py` | `HybridRetriever`: orchestrates BM25 + ChromaDB + RRF fusion + metadata filtering; accepts `chroma_persist_dir` for persistent vector storage |
| `bm25_engine.py` | `BM25Engine`: BM25Okapi indexing + retrieval with jieba trigram tokenization (Chinese segments 2-8 chars → trigrams for segments ≥4 chars) |
| `vector_engine.py` | `VectorEngine`: ChromaDB ANN search (HNSW, cosine) + brute-force fallback + Jaccard fallback; content-hash MD5 document IDs |
| `fusion.py` | `rrf_fusion()`, `hybrid_fusion()`: RRF and weighted score fusion. MD5-based stable document IDs for deduplication |
| `filters.py` | `apply_filters()`: metadata-based filtering (source, date, doc_type) |
| `chunker.py` | `TextChunker`: document splitting with overlap + metadata tagging |
| `preprocessor.py` | `TextPreprocessor` (cleaning, boilerplate removal, paragraph dedup — enabled by default), `RelevanceGate` (relevance filtering), `DocTypeClassifier` (fast classification) |
| `query_parser.py` | `QueryParser`: intent detection, entity extraction, date parsing from queries |
| `dictionaries.py` | Externalized keyword dictionaries: `STOCK_MAP`, `FINANCIAL_TERMS`, `INDUSTRY_TERMS`, etc. |
| `persistence.py` | `save_index()`, `load_index()`: index serialization |

### `financial_rag/llm/` — LLM Layer

| File | Role |
|------|------|
| `dashscope_client.py` | DashScope API: LLM + Embedding + Rerank |
| `model_router.py` | Auto-select model by task complexity + budget control (4 tiers), `get_caller()` / `get_caller_for_agent()` |
| `caller.py` | `LLMCaller`: retry + balanced-bracket JSON parsing + response cache + input validation + anti-hallucination constraints. Used by tools, SlotFiller, and pipeline |

### `financial_rag/guard/` — Anti-Hallucination

| File | Role |
|------|------|
| `reflector.py` | `HallucinationGuard`: 4-layer check — L1 source grounding (jieba token overlap), L2 numerical fidelity (number+unit pairs), L3 citation integrity ([N] references valid), L4 structure compliance (expected sections). Also `ReflectionLoop` (ReAct) |

### `financial_rag/services/` — Business Logic Layer

| File | Role | Key exports |
|------|------|-------------|
| `analysis.py` | Pure analysis functions (no HTTP deps, DI via kwargs) | `analyze_news_text()`, `analyze_topic_research()`, `_extract_confidence()`, `_parse_verdict()` |
| `persistence.py` | KB / Meta / Archive JSON read/write + index persistence | `load_kb()`, `save_kb()`, `save_index()`, `load_index()`, `append_news_archive()` |

### `financial_rag/static/` — Frontend

| File | Role |
|------|------|
| `index.html` | Web UI structure (273 lines) |
| `styles.css` | Dark theme styling (209 lines) |
| `app.js` | Frontend logic + API interaction (920 lines) |

---

## CLI Quick Reference

```bash
# Web UI (recommended)
python -m financial_rag.main web

# Pipeline (auto-routed by AgentRouter)
python -m financial_rag.main pipeline "商汤科技2024年营收增长"
python -m financial_rag.main pipeline "茅台走势" -v              # → kline chain
python -m financial_rag.main pipeline "2024-06-01 发生了什么"    # → event_impact chain

# Function Calling
python -m financial_rag.main toolcall -l                  # list all 27 tools
python -m financial_rag.main toolcall "商汤科技营收增长" -v

# News / KLine / Slot / Score
python -m financial_rag.main news "AI大模型" -s
python -m financial_rag.main kline "人工智能ETF" --days 30 -s
python -m financial_rag.main slot "智谱AI融资分析" -t fin
python -m financial_rag.main score "商汤科技营收" -k 5
```

Template options: `quick` (default) | `fin` | `news` | `deep`

---

## Configuration

All config in `financial_rag/config.py`. Global instance: `from financial_rag.config import config`.

| Section | Key params | Defaults |
|---------|-----------|----------|
| `config.llm` | model, embedding_model, rerank_model, temperature | qwen-plus, text-embedding-v3, qwen3-rerank, 0.0 |
| `config.coordinator` | execution_mode, max_parallel_agents, max_retries | sequential, 3, 1 |
| `config.pipeline` | hybrid_top_k, rrf_k, bm25_weight, vector_weight | 10, 60, 0.3, 0.7 |
| `config.reflection` | max_retrievals, max_steps, min_confidence | 3, 6, 0.6 |

Env vars (`.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `DASHSCOPE_API_KEY` | No | Alibaba DashScope API key. Without it → local-only mode |
| `TUSHARE_TOKEN` | No | Tushare Pro API token. 120+ points for `daily` endpoint |
| `MOCK_MODE` | No | `true` → data sources return mock data (LLM always real) |

---

## Debug Guide

| Problem | Where to look |
|---------|---------------|
| Wrong agent chain selected | `core/agent_router.py` `_classify_intent()`. Check keyword weights and confidence threshold (0.5) |
| Agent not working | `core/orchestrator.py` `_apply_updates()`. Set `verbose = True` |
| Upstream data lost | Verify `_apply_updates()` merge semantics — dict attributes must merge, not replace |
| K-line fetch fails | Check `.env` has `TUSHARE_TOKEN` with 120+ points |
| News fetch fails | `from financial_rag.rss_fetcher import fetch_all_news; fetch_all_news()` |
| Retrieval inaccurate | `python -m financial_rag.main score "query"`. Adjust `config.pipeline` weights |
| LLM hallucination | Check `guard/reflector.py` HallucinationGuard (4-layer check). Lower `temperature` |
| Adding a new Agent | Inherit `BaseAgent`, implement `process()`, register in `factory.py` |
| Adding a new intent | Register in `AgentRouter.register_intent()` — keywords + chain mapping |

---

## Programming API

```python
# LLM call
from financial_rag.llm import get_llm
llm = get_llm(api_key=config.llm.api_key, model="qwen-plus")
resp = llm.chat("分析商汤科技2024年营收增长")

# Hybrid retrieval
from financial_rag.retrievers import HybridRetriever
retriever = HybridRetriever(embedder=get_embedding(), reranker=get_reranker())
retriever.index(docs)
results = retriever.search("商汤科技营收", top_k=5)

# Agent routing
from financial_rag.core.agent_router import AgentRouter
router = AgentRouter()
decision = router.route("茅台走势")
print(decision.intent)   # "kline"
print(decision.chain)    # ["AnalysisAgent", "ScoringAgent"]

# Function Calling
from financial_rag.tools import create_financial_registry, create_tool_session
registry = create_financial_registry()
session = create_tool_session(llm=get_llm(), registry=registry)
stats = session.run("商汤科技营收增长多少")

# Analysis service (pure functions)
from financial_rag.services.analysis import analyze_news_text, analyze_topic_research
result = analyze_news_text("商汤科技2024年营收50.3亿元，同比增长36.4%")
print(result["assessment"])  # bullish / bearish / neutral
```
