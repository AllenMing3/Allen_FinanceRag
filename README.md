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
| **7 Specialized Agents** | Coordinator · Ingestion · Extraction · Report · KLine · EventImpact · Scoring — each focused on a single responsibility |
| **Agent = Orchestrator** | Agents contain zero business logic; all work is delegated through `call_tool()` — a strict separation enforced by architecture rule |
| **26 Registered Tools** | Function-calling tools across 7 modules (extraction, news, kline, event_impact, scoring, coordinator, report) |
| **Hybrid Retrieval** | BM25 + 1024-dim vector embedding + RRF fusion + qwen3-rerank + TextChunker with metadata filtering |
| **Full-Chain Scoring** | Every query ends with `ScoringAgent` — pipeline quality evaluation, hallucination guard, and structured score report |
| **LLM-Optional** | Heuristic fallback when no API key is configured; mock mode for data sources enables fully offline development |
| **234 Unit Tests** | All pass in < 0.2 s, no API key required, regex-fallback for extraction tools |

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
│                    Agent Layer (7 agents)                    │
│   Coordinator · Ingestion · Extraction · Report             │
│   KLine · EventImpact · Scoring                             │
│   ─────────────────────────────────────                     │
│   All agents delegate via call_tool() — no business logic   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Tool Layer (26 tools)                     │
│   extraction · news · kline · event_impact                  │
│   scoring · coordinator · report                            │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
┌──────────▼───┐  ┌──────▼──────┐  ┌───▼────────────────┐
│  Retrieval   │  │  Data APIs  │  │  DashScope Qwen    │
│  BM25+Vector │  │  Tushare    │  │  LLM · Embedding   │
│  + RRF       │  │  10jqka     │  │  · Rerank          │
│  + Rerank    │  │  Sina/East  │  │                    │
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
| `kline` | "茅台走势", "RSI指标", "支撑位" | KLineAgent → ReportAgent → ScoringAgent |
| `event_impact` | "利好利空", "并购重组", "涨停分析" | EventImpactAgent → ReportAgent → ScoringAgent |
| `report` | "财报分析", "营收增长", "年报" | IngestionAgent → ExtractionAgent → ReportAgent → ScoringAgent |
| `news` | "最新动态", "行业新闻" | IngestionAgent → ReportAgent → ScoringAgent |
| `general` | fallback | IngestionAgent → ExtractionAgent → ReportAgent → ScoringAgent |

**Design principle:** Every chain ends with `ScoringAgent` to ensure output quality. `CoordinatorAgent` is never placed in chains — the pipeline itself handles routing.

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
| Retrieval | Custom BM25 + Vector + RRF fusion + TextChunker + metadata filter |
| Testing | pytest — 234 tests, regex-fallback for offline operation |

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
python -m financial_rag.main toolcall -l                     # list all 26 tools
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
| `test_new_agents.py` | Coordinator / KLine / EventImpact / Scoring / Report agents | 29 |
| `test_mock_data.py` | Mock K-line, news search, financial indicators | 31 |
| `test_agents.py` | IngestionAgent + ExtractionAgent + full agent chain | 22 |
| `test_new_tools.py` | Scoring / coordinator / report / event_impact tools | 22 |
| `test_analysis.py` | News analysis + topic research (mock mode) | 22 |
| `test_analysis_tools.py` | Growth rate, ratio, compare, summarize | 16 |
| `test_factory.py` | 7-agent factory wiring, chain ordering | 12 |
| `test_orchestrator_merge.py` | Metadata merge, findings extend, scalar replace | 10 |
| **Total** | | **234** |

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
├── agents/          # 7 specialized agents (all delegate via call_tool)
│   ├── coordinator_agent.py
│   ├── ingestion_agent.py
│   ├── extraction_agent.py
│   ├── report_agent.py
│   ├── kline_agent.py
│   ├── event_impact_agent.py
│   └── scoring_agent.py
├── core/            # Architecture engine
│   ├── agent_router.py    # Intent classification + chain selection
│   ├── orchestrator.py    # Multi-agent scheduling (dict merge / list extend)
│   ├── pipeline.py        # 5-phase PipelineScheduler
│   ├── factory.py         # create_orchestrator, setup_environment
│   └── scorer.py          # PipelineScoreCard
├── tools/           # 26 registered tools across 7 modules
│   ├── core.py            # FunctionRegistry, ToolExecutor, ToolCallSession
│   ├── extraction_tools.py
│   ├── news_tools.py
│   ├── kline_tools.py     # Also hosts STOCK_MAP (shared data)
│   ├── event_impact_tools.py
│   ├── scoring_tools.py
│   ├── coordinator_tools.py
│   └── report_tools.py
├── retrievers/      # HybridRetriever: BM25 + Vector + RRF + TextChunker
├── llm/             # DashScope client + model router (4 tiers)
├── static/          # Frontend (HTML / CSS / JS, dark theme)
├── config.py        # Global configuration (LLM, RAG, Coordinator, Pipeline)
├── web.py           # FastAPI endpoints
└── main.py          # CLI entry (argparse)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Agent = Orchestrator only** | Agents call `call_tool()` — never import business modules directly. Enforced by `agent-tool-delegation.md` rule |
| **Metadata merge (not replace)** | `orchestrator._apply_updates()` merges dicts and extends lists, preventing downstream agents from wiping upstream data |
| **CoordinatorAgent excluded from chains** | Pipeline's own `_route_query()` already handles routing; adding CoordinatorAgent would create a redundant second router |
| **STOCK_MAP lives in `kline_tools.py`** | Shared data belongs in the tool layer — agents import from tools, never from each other |
| **ScoringAgent on every chain** | Guarantees quality evaluation and hallucination check on every output |

---

## License

MIT
