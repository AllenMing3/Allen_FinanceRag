# Financial RAG — AI Sector Intelligent Analysis System

A Retrieval-Augmented Generation (RAG) system for the **AI/technology sector**. Multi-agent orchestration, intent-based routing, function calling, and hybrid retrieval — powered by DashScope (Qwen).

> 📖 [Architecture Docs](docs/ARCHITECTURE.md) · 📘 [User Guide (中文)](USER_GUIDE.md) · 📋 [Interview Q&A (中文)](docs/PROJECT_QA.md)

---

## Screenshots

<p align="center">
  <img src="docs/screenshots/agent_architecture.png" alt="Agent Architecture" width="100%">
  <em>4-Agent chain: CoordinatorAgent → IngestionAgent → AnalysisAgent → ScoringAgent</em>
</p>

<p align="center">
  <img src="docs/screenshots/analysis.png" alt="Analysis" width="100%">
  <em>News interpretation, topic research, and accumulated learning history</em>
</p>

<p align="center">
  <img src="docs/screenshots/kb_management.png" alt="Knowledge Base" width="100%">
  <em>KB status, hybrid indexing (BM25 + Embedding + RRF), and keyword-based management</em>
</p>

---

## Why This System

Traditional RAG pipelines treat every query the same way. Financial analysis doesn't work like that — a K-line technical analysis, an event impact assessment, and an earnings report each require **different data sources, different agent chains, and different evaluation criteria**.

This system solves that with **intent-aware routing**: "茅台走势" triggers K-line analysis with MACD/RSI/Bollinger, while "2024-06-01 发生了什么" triggers event impact scoring — without the user specifying which pipeline to use.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Intent Routing** | AgentRouter classifies queries across 5 intents (kline / event_impact / report / news / general) with dynamic agent chain selection |
| **4 Agents** | Coordinator · Ingestion · Analysis · Scoring — agents call `call_tool()`, never business logic directly |
| **28 Tools** | Function-calling tools across 9 modules — all heavy computation lives in tools, not agents |
| **Hybrid Retrieval** | BM25 + 1024-dim vector + RRF fusion + qwen3-rerank + metadata filtering |
| **LLMCaller** | Retry + JSON parsing + caching + anti-hallucination constraints on every LLM call |
| **Full-Chain Scoring** | Every chain ends with ScoringAgent — pipeline quality evaluation + hallucination guard (6-layer check) |
| **507 Tests** | All pass in < 5 s, no API key required, mock data sources only |

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
│   AgentRouter (intent → chain) · PipelineScheduler (5-phase)│
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Agent Layer (4 agents)                    │
│   Coordinator · Ingestion · Analysis · Scoring             │
│   AnalysisAgent routes by intent via call_tool()            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌──────────┬──────────────┬───────────────────────────────────┐
│ Retrieval│  Data APIs   │  LLM Layer                        │
│ BM25+Vec │  Tushare     │  LLMCaller (retry+cache+JSON)     │
│ +RRF+Rerank│ 10jqka/East│  ModelRouter (4 tiers, Qwen)      │
└──────────┴──────────────┴───────────────────────────────────┘
```

**Pipeline:** Fetch → Index → Process → Output → Evolve

---

## Agent Routing

| Intent | Examples | Agent Chain |
|--------|----------|-------------|
| `kline` | "茅台走势", "RSI指标" | AnalysisAgent → ScoringAgent |
| `event_impact` | "利好利空", "并购重组" | AnalysisAgent → ScoringAgent |
| `report` | "财报分析", "营收增长" | Ingestion → Analysis → Scoring |
| `news` | "最新动态", "行业新闻" | Ingestion → Analysis (deep) → Scoring |
| `general` | fallback | Ingestion → Analysis → Scoring |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | DashScope Qwen (turbo / plus / max / 235b) |
| Embedding | text-embedding-v3 (1024-dim) |
| Rerank | qwen3-rerank (fallback to RRF) |
| Backend | FastAPI + 4 async routers |
| Data APIs | Tushare, 10jqka, Sina, EastMoney |
| Retrieval | BM25 + Vector + RRF + TextChunker + QueryParser |
| Testing | pytest — 507 tests across 21 files |

---

## Quick Start

```bash
# Setup
python -m venv myenv && source myenv/bin/activate
# Windows: .\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # fill DASHSCOPE_API_KEY

# Launch Web UI
python -m financial_rag.main web
# → http://127.0.0.1:8000

# CLI examples
python -m financial_rag.main pipeline "商汤科技2024年营收增长"
python -m financial_rag.main pipeline "茅台走势" -v          # auto-routes to K-line
python -m financial_rag.main toolcall -l                     # list all 28 tools
```

---

## Testing

```bash
python -m pytest tests/ -v    # 507 tests, < 5s, no API key needed
```

Tests mock only **data sources** (Tushare, news APIs). LLM, embedding, and rerank stay real. Extraction tools have regex fallback for offline operation.

---

## License

MIT
