# Architecture Deep Dive

> This document provides detailed architecture documentation. See [README.md](../README.md) for project overview and quick start.

## Component Responsibilities

| Engine | File | Responsibility |
|--------|------|----------------|
| **Route** | `core/agent_router.py` | Intent classification (5 domains), agent chain selection, metadata extraction (date/stock) |
| **Coordinate** | `core/orchestrator.py` | Register agents, decide execution order, pass context. Metadata merge (not replace), list extend |
| **Schedule** | `core/pipeline.py` | 5-phase pipeline: Fetch → Index → Process (AgentRouter) → Output → Evolve |
| **Indexer** | `core/indexer.py` | 4-stage retrieval: Clean → Extract → Retrieve → Verify. BM25 + Vector + RRF fusion |
| **Reflect** | `core/reflector.py` | ReAct loop (Think → Act → Observe → Judge) + 6-layer anti-hallucination guard |
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
     injection       │ BM25  │ │Vector │
                     │ Index │ │1024-d │
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
| `kline` | KLineAgent → ReportAgent → ScoringAgent |
| `event_impact` | EventImpactAgent → ReportAgent → ScoringAgent |
| `report` | IngestionAgent → ExtractionAgent → ReportAgent → ScoringAgent |
| `news` | IngestionAgent → ReportAgent → ScoringAgent |
| `general` | IngestionAgent → ExtractionAgent → ReportAgent → ScoringAgent |

**Low-confidence override:** When intent confidence < 0.5, `IngestionAgent` is prepended to ensure context gathering before downstream agents.

### Metadata Extraction

Alongside intent classification, `AgentRouter` extracts structured metadata:

- **Date**: `YYYY-MM-DD`, `YYYY年M月D日`, `YYYYMMDD` → stored in `routing_decision.metadata.date`
- **Stock**: keyword lookup via `STOCK_MAP` (11 stocks), SH/SZ code patterns → stored in `routing_decision.metadata.ts_code`
- **Context override**: If query already contains `ts_code` in orchestrator context, it is preserved

---

## Agent Chain (Phase 3 Detail)

Agents are **lightweight orchestrators** — all business logic delegated to registered tools via Function Calling.

### Core Extraction Chain (report / general)

```
IngestionAgent
  → call_tool(extract_document_metadata)
  → call_tool(detect_document_type)
  → context →

ExtractionAgent
  → call_tool(extract_financial_metrics)
  → call_tool(extract_entities)
  → call_tool(generate_search_queries)
  → context →

ReportAgent
  → call_tool(synthesize_report)
  → context →

ScoringAgent
  → call_tool(evaluate_pipeline_quality)
  → call_tool(check_hallucination)
  → call_tool(generate_score_report)
```

### K-Line Chain

```
KLineAgent
  → call_tool(fetch_kline_report)     # STOCK_MAP resolve + Tushare fetch
  → call_tool(analyze_kline)          # MACD / RSI / Bollinger / KDJ
  → call_tool(generate_kline_analysis) # LLM interpretation
  → context →

ReportAgent → ScoringAgent
```

### Event Impact Chain

```
EventImpactAgent
  → call_tool(fetch_date_events)       # Date-based event retrieval
  → call_tool(assess_event_impact)     # Bullish / bearish + impact factor
  → context →

ReportAgent → ScoringAgent
```

### AI Industry Metrics (ExtractionAgent)

| Category | Metrics |
|----------|---------|
| Financial | revenue, net_income, gross_margin, rd_expense, arr |
| Compute | gpu_count, training_cluster_size, inference_cost_per_token, compute_utilization |
| Model | model_params, context_window, inference_latency, benchmark_score |
| Commercial | api_calls, customer_count, dau, mau |

**Entity extraction dimensions:** companies, persons, ai_models, chips_hardware, tech_terms, financial_figures, event, industries

---

## Orchestrator Metadata Merge

`orchestrator._apply_updates()` uses **merge semantics** — not wholesale replacement — to prevent downstream agents from wiping upstream data:

| Update Type | Behavior | Example |
|-------------|----------|---------|
| **Dict attribute** (e.g. `metadata`) | `current.update(v)` — merge, not replace | KLineAgent adds `ts_code`, ReportAgent adds `report_type` → both preserved |
| **List attribute** (e.g. `intermediate_findings`) | `current.extend(v)` — append, not replace | Each agent's findings accumulate across the chain |
| **Scalar attribute** (e.g. `final_answer`) | `setattr(self.context, k, v)` — direct replace | Last agent's answer wins |
| **Unknown key** | Written to `context.metadata` | Custom fields from any agent land in metadata |

---

## Retrieval Modes

| Mode | Condition | Chain |
|------|-----------|-------|
| Local only | No API Key | BM25 + Jaccard → RRF |
| With Embedding | API Key set | BM25 + Vector (1024-dim) → RRF |
| Full pipeline | API Key active | BM25 + Vector → RRF → qwen3-rerank |

`HybridRetriever` also applies `TextChunker` (split + overlap + metadata tagging) at index time and metadata filtering at query time.

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
| `slot_filler.py` | Parallel slot filling engine with TTFT measurement | `SlotFiller`, `create_slot_filler` |
| `rss_fetcher.py` | Financial news via domestic APIs (10jqka / Sina / EastMoney) + rate limiting | `search_news`, `fetch_all_news` |
| `tushare_client.py` | K-line & financial indicators via Tushare Pro | `fetch_stock_kline`, `compute_technical_indicators` |
| `mock_data.py` | 25 AI-sector mock news + 3 long-form articles | Mock data for offline dev |
| `web.py` | FastAPI Web UI server — thin shell, delegates to `services/` | FastAPI app, `/api/*` endpoints |

### `financial_rag/core/` — Architecture Layer

| File | Role | Key exports |
|------|------|-------------|
| `base.py` | Abstract foundations | `BaseAgent`, `AgentContext`, `AgentResult`, `ExecutionMode` |
| `agent_router.py` | Query-time routing: intent classification, chain selection, metadata extraction | `AgentRouter`, `RoutingDecision` |
| `orchestrator.py` | Multi-agent scheduling engine — dict merge, list extend, scalar replace | `AgentOrchestrator` |
| `pipeline.py` | 5-phase PipelineScheduler (Fetch → Index → Process → Output → Evolve) | `PipelineScheduler`, `PipelineResult` |
| `router.py` | CLI command dispatch + handlers | `CommandRouter` |
| `factory.py` | Factory: creates and wires 7 agents + AgentRouter | `create_orchestrator`, `setup_environment` |
| `indexer.py` | Hybrid retrieval pipeline orchestration | `PipelineOrchestrator` |
| `reflector.py` | ReAct loop + HallucinationGuard | `ReflectionLoop`, `HallucinationGuard` |
| `scorer.py` | Full-pipeline scorecard | `PipelineScoreCard`, `ScoreGrade` |
| `protocol.py` | Agent messaging infrastructure | `AgentMessage`, `MessageBus` |

### `financial_rag/agents/` — 7 Agents

| File | Role |
|------|------|
| `coordinator_agent.py` | Intent classification + chain selection via `call_tool(classify_query_intent, select_agent_chain)` |
| `ingestion_agent.py` | Data ingestion → `call_tool(extract_document_metadata, detect_document_type)` |
| `extraction_agent.py` | Feature extraction → `call_tool(extract_financial_metrics, extract_entities, generate_search_queries)` |
| `report_agent.py` | LLM synthesis → `call_tool(synthesize_report)` — converts findings to structured documents |
| `kline_agent.py` | K-line analysis → `call_tool(fetch_kline_report, analyze_kline, generate_kline_analysis)` |
| `event_impact_agent.py` | Event impact → `call_tool(fetch_date_events, assess_event_impact)` |
| `scoring_agent.py` | Quality scoring → `call_tool(evaluate_pipeline_quality, check_hallucination, generate_score_report)` |
| `utils.py` | Shared: `build_news_context()` |

### `financial_rag/tools/` — 26 Registered Tools across 8 Modules

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
| `__init__.py` | — | `create_financial_registry()` — registers all 26 tools; re-exports `STOCK_MAP` |

### `financial_rag/retrievers/` — Hybrid Retrieval

| File | Role |
|------|------|
| `retriever.py` | `HybridRetriever`: BM25 + Vector + RRF fusion + metadata filtering |
| `chunker.py` | `TextChunker`: document splitting with overlap + metadata tagging |

### `financial_rag/llm/` — LLM Layer

| File | Role |
|------|------|
| `dashscope_client.py` | DashScope API: LLM + Embedding + Rerank |
| `model_router.py` | Auto-select model by task complexity + budget control (4 tiers) |

### `financial_rag/services/` — Business Logic Layer

| File | Role | Key exports |
|------|------|-------------|
| `analysis.py` | Pure analysis functions (no HTTP deps, DI via kwargs) | `analyze_news_text()`, `analyze_topic_research()` |
| `persistence.py` | KB / Meta / Archive JSON read / write | `load_kb()`, `save_kb()`, `load_meta()`, `save_meta()`, `append_news_archive()` |

### `financial_rag/static/` — Frontend

| File | Role |
|------|------|
| `index.html` | Web UI structure (344 lines) |
| `styles.css` | Dark theme styling (175 lines) |
| `app.js` | Frontend logic + API interaction (426 lines) |

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
python -m financial_rag.main toolcall -l                  # list all 26 tools
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
| `config.coordinator` | execution_mode, max_parallel_agents, max_retries | sequential, 3, 2 |
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
| LLM hallucination | Check `core/reflector.py` HallucinationGuard. Lower `temperature` |
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
print(decision.chain)    # ["KLineAgent", "ReportAgent", "ScoringAgent"]

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
