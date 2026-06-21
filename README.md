# Financial RAG — AI Sector Intelligent Analysis System

A Retrieval-Augmented Generation (RAG) system purpose-built for the **AI/technology sector**. It combines multi-agent orchestration, intent-based routing, function calling, and hybrid retrieval to deliver structured financial and technical analysis — all powered by Alibaba DashScope (Qwen).

> 📖 Detailed architecture docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)  
> 📘 User guide (中文): [USER_GUIDE.md](USER_GUIDE.md)

---

## Why This System

Traditional RAG pipelines treat every query the same way. Financial analysis doesn't work like that — a K-line technical analysis, an event impact assessment, and a quarterly earnings report each require **different data sources, different agent chains, and different evaluation criteria**.

This system solves that with **intent-aware routing**: a query like "茅台走势" automatically triggers K-line analysis with MACD/RSI/Bollinger indicators, while "2024-06-01 发生了什么" triggers event impact scoring — without the user ever specifying which pipeline to use.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **AgentRouter** | Automatic query intent classification across 5 domains (kline / event_impact / report / news / general), with dynamic agent chain selection |
| **4 Streamlined Agents** | Coordinator · Ingestion · Analysis · Scoring — AnalysisAgent merges extraction, K-line, event impact, and report generation into a single intent-driven unit |
| **LLMCaller Protection** | Retry (exponential backoff) + structured JSON parsing + response caching + input validation + anti-hallucination constraints — wraps every LLM call |
| **28 Registered Tools** | Function-calling tools across 9 modules (extraction, news, kline, event_impact, scoring, coordinator, report, analysis, core) |
| **Hybrid Retrieval** | BM25 + 1024-dim vector embedding + RRF fusion + qwen3-rerank + TextChunker + TextPreprocessor + QueryParser + DataOrchestrator (multi-pool routing) |
| **Full-Chain Scoring** | Every query ends with `ScoringAgent` — pipeline quality evaluation, hallucination guard, and structured score report |
| **LLM-Optional** | Heuristic fallback when no API key is configured; mock mode for data sources enables fully offline development |
| **369 Unit Tests** | All pass in < 5 s, no API key required, regex-fallback for extraction tools |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Layer                             │
│   CLI · FastAPI Web UI (dark theme)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Scheduling Layer                          │
│   CommandRouter → PipelineScheduler (5-phase)               │
│   AgentRouter (intent classification + chain selection)     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Agent Layer (4 agents)                    │
│   Coordinator · Ingestion · Analysis · Scoring             │
│   ─────────────────────────────────────                     │
│   AnalysisAgent routes by intent: extraction / kline /      │
│   event_impact / report — all via call_tool()               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Tool Layer (28 tools)                     │
│   extraction · news · kline · event_impact                  │
│   scoring · coordinator · report · analysis                 │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
┌──────────▼───┐  ┌──────▼──────┐  ┌───▼────────────────┐
│  Retrieval   │  │  Data APIs  │  │  LLM Call Layer    │
│  BM25+Vector │  │  Tushare    │  │  LLMCaller (retry  │
│  + RRF       │  │  10jqka     │  │  + cache + JSON)   │
│  + Rerank    │  │  Sina/East  │  │  ModelRouter (4T)  │
│  DataOrch.   │  │             │  │  DashScope Qwen    │
└──────────────┘  └─────────────┘  └────────────────────┘
```

### 5-Phase Pipeline

```
Fetch → Index → Process → Output → Evolve
         │        │
         │        └─ AgentRouter selects agent chain by query intent
         └─ BM25 + Vector indexing with TextChunker
```

---

## Agent Routing

`AgentRouter` classifies query intent using keyword and pattern matching, then selects the optimal agent chain:

| Intent | Trigger Examples | Agent Chain |
|--------|-----------------|-------------|
| `kline` | "茅台走势", "RSI指标", "支撑位" | AnalysisAgent (intent=kline) → ScoringAgent |
| `event_impact` | "利好利空", "并购重组", "涨停分析" | AnalysisAgent (intent=event_impact) → ScoringAgent |
| `report` | "财报分析", "营收增长", "年报" | IngestionAgent → AnalysisAgent (intent=general) → ScoringAgent |
| `news` | "最新动态", "行业新闻" | IngestionAgent → AnalysisAgent (intent=news, deep analysis) → ScoringAgent |
| `general` | fallback | IngestionAgent → AnalysisAgent (intent=general) → ScoringAgent |

**Design principle:** Every chain ends with `ScoringAgent` to ensure output quality. `CoordinatorAgent` is never placed in chains — the pipeline itself handles routing. `AnalysisAgent` consolidates extraction, K-line, event impact, and report generation — selecting tools by `intent` metadata.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Alibaba DashScope (Qwen family: turbo / plus / max / 235b) |
| Embedding | text-embedding-v3 (1024-dim) |
| Rerank | qwen3-rerank (graceful fallback to RRF if unavailable) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |
| Data APIs | Tushare Pro, 10jqka, Sina Finance, EastMoney |
| Retrieval | Custom BM25 + Vector + RRF fusion + TextChunker + TextPreprocessor + DataOrchestrator + metadata filter |
| Testing | pytest — 369 tests, regex-fallback for offline operation |

---

## Quick Start

```bash
# 1. Setup
python -m venv myenv && source myenv/bin/activate
# Windows: .\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # fill DASHSCOPE_API_KEY (optional for local mode)

# 2. Launch Web UI
python -m financial_rag.main web
# → http://127.0.0.1:8000

# 3. CLI examples
python -m financial_rag.main pipeline "商汤科技2024年营收增长"
python -m financial_rag.main pipeline "茅台走势" -v          # auto-routes to K-line
python -m financial_rag.main pipeline "2024-06-01 发生了什么" # auto-routes to event impact
python -m financial_rag.main toolcall -l                     # list all tools
```

---

## Domain Focus

The system is calibrated for the **AI/technology sector**:

- **11 document types** — annual reports, quarterly filings, announcements, policy, news, research reports, technical reports, product launches, funding rounds, industry analysis, other
- **12 core metrics** — financial (revenue, net_income, gross_margin) + compute (gpu_count, inference_cost) + model (params, latency, benchmark_score) + commercial (api_calls, dau, customer_count)
- **9 entity dimensions** — company, person, AI model, chip/hardware, tech term, financial figure, event, industry, topic

---

## Testing

```bash
python -m pytest tests/ -v
```

| Test File | Coverage | Tests |
|-----------|----------|------:|
| `test_agent_router.py` | Intent classification, chain selection, metadata extraction | 36 |
| `test_extraction_tools.py` | 5 extraction tools (regex fallback) + long-article extraction | 34 |
| `test_new_agents.py` | Coordinator / Analysis / Scoring agents | 29 |
| `test_mock_data.py` | Mock K-line, news search, financial indicators | 31 |
| `test_agents.py` | IngestionAgent + AnalysisAgent + full agent chain | 22 |
| `test_new_tools.py` | Scoring / coordinator / report / event_impact tools | 22 |
| `test_analysis.py` | News analysis + topic research (mock mode) | 22 |
| `test_analysis_tools.py` | Growth rate, ratio, compare, summarize | 16 |
| `test_llm_caller.py` | LLMCaller retry, JSON, cache, constraints | 38 |
| `test_data_orchestrator.py` | DataOrchestrator multi-pool ingest/search/cross-search | 27 |
| `test_query_parser.py` | QueryParser intent, entity, date extraction | 23 |
| `test_factory.py` | 4-agent factory wiring, chain ordering | 12 |
| `test_orchestrator_merge.py` | Metadata merge, findings extend, scalar replace | 10 |
| `test_smoke.py` | Web API smoke tests (all endpoints) | 55 |
| `test_persistence.py` | Index save/load, dedup, backup rotation | 32 |
| **Total** | | **369** |

Tests mock only data sources — LLM, embedding, and rerank stay real. Extraction tools use regex fallback, so no API key is needed.

---

## Mock Mode

Set `MOCK_MODE=true` in `.env` for offline development:

| Source | Behavior |
|--------|----------|
| K-line data | 8 stocks + 7 ETFs with geometric Brownian motion simulation |
| News API | 25 built-in AI-sector news articles + 3 long-form documents |
| LLM / Embedding / Rerank | **Always real** DashScope API (requires key) |

An orange badge appears in the Web UI when mock mode is active.

---

## Project Structure

```
financial_rag/
├── agents/          # 4 streamlined agents (all delegate via call_tool)
│   ├── coordinator_agent.py   # Intent classification + chain selection
│   ├── ingestion_agent.py     # Data ingestion + metadata extraction
│   ├── analysis_agent.py      # Unified analysis (extraction + kline + event + report)
│   └── scoring_agent.py       # Pipeline scoring + hallucination check
├── core/            # Architecture engine
│   ├── agent_router.py        # Intent classification + chain selection
│   ├── orchestrator.py        # Multi-agent scheduling (dict merge / list extend)
│   ├── data_orchestrator.py   # Multi-pool text routing (KnowledgePool + DataRouter)
│   ├── pipeline.py            # 5-phase PipelineScheduler
│   ├── factory.py             # create_orchestrator, setup_environment
│   └── scorer.py              # PipelineScoreCard
├── tools/           # 28 registered tools across 9 modules
│   ├── core.py                # FunctionRegistry, ToolExecutor, ToolCallSession
│   ├── extraction_tools.py
│   ├── news_tools.py
│   ├── kline_tools.py         # Also hosts STOCK_MAP (shared data)
│   ├── event_impact_tools.py
│   ├── scoring_tools.py
│   ├── coordinator_tools.py
│   ├── report_tools.py
│   └── analysis_tools.py      # Deep analysis: analyze_news_deep, analyze_topic_deep
├── retrievers/      # Modular retrieval stack
│   ├── retriever.py           # HybridRetriever: BM25 + Vector + RRF
│   ├── bm25_engine.py         # BM25 scoring engine
│   ├── vector_engine.py       # Vector similarity engine
│   ├── fusion.py              # RRF + hybrid fusion
│   ├── filters.py             # Metadata filtering
│   ├── chunker.py             # TextChunker: split + overlap + metadata
│   ├── preprocessor.py        # TextPreprocessor + RelevanceGate + DocTypeClassifier
│   ├── query_parser.py        # QueryParser: intent + entity + date extraction
│   ├── dictionaries.py        # Externalized keyword dictionaries
│   └── persistence.py         # Index save/load
├── llm/             # DashScope client + model router + LLMCaller
│   ├── dashscope_client.py    # DashScope API: LLM + Embedding + Rerank
│   ├── model_router.py        # Auto-select model by task complexity (4 tiers)
│   └── caller.py              # LLMCaller: retry + JSON + cache + constraints
├── guard/           # Anti-hallucination guard
│   └── reflector.py           # HallucinationGuard (6-layer check)
├── static/          # Frontend (HTML / CSS / JS, dark theme)
├── config.py        # Global configuration (LLM, RAG, Coordinator, Pipeline)
├── web.py           # FastAPI endpoints
└── main.py          # CLI entry (argparse)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Agent = Orchestrator only** | Agents call `call_tool()` — never import business modules directly. Enforced by architecture rule |
| **AnalysisAgent consolidation** | 5 agents (Extraction, KLine, EventImpact, Report + intent) merged into 1 — `context.metadata["intent"]` selects tool chain, reducing chain complexity |
| **Deep analysis via tools** | `analysis_tools.py` wraps `services/analysis.py` so agent chain produces same structured output (multi-dim impact, key signals, sub-topics) as direct API endpoints |
| **Metadata merge (not replace)** | `orchestrator._apply_updates()` merges dicts and extends lists, preventing downstream agents from wiping upstream data |
| **CoordinatorAgent excluded from chains** | Pipeline's own `_route_query()` already handles routing; adding CoordinatorAgent would create a redundant second router |
| **LLMCaller wrapping** | All LLM calls go through `LLMCaller` for retry, JSON parsing, caching, and anti-hallucination constraints — no bare `llm.chat()` in tools |
| **STOCK_MAP lives in `kline_tools.py`** | Shared data belongs in the tool layer — agents import from tools, never from each other |
| **ScoringAgent on every chain** | Guarantees quality evaluation and hallucination check on every output |

---

## License

MIT
