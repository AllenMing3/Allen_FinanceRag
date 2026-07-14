# FinRAG — Intent-Aware Multi-Agent RAG System

**Not all queries are equal.** A stock price question needs K-line data. A breaking-news question needs event impact scoring. An earnings question needs report synthesis. FinRAG routes each query to the right agent chain automatically — no manual pipeline selection.

![Tests](https://img.shields.io/badge/tests-609%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-gray)

> [Architecture Docs](docs/ARCHITECTURE.md) · [User Guide (中文)](USER_GUIDE.md) · [Interview Q&A (中文)](docs/PROJECT_QA.md)

---

<p align="center">
  <img src="docs/screenshots/agent_architecture.png" alt="Agent Architecture" width="100%">
  <em>4-agent chain: Coordinator → Ingestion → Analysis → Scoring, with intent-based routing</em>
</p>

<p align="center">
  <img src="docs/screenshots/analysis.png" alt="Smart Query" width="100%">
  <em>RAG retrieval with hybrid search + K-line technical analysis in one unified query panel</em>
</p>

<p align="center">
  <img src="docs/screenshots/kb_management.png" alt="Data Management" width="100%">
  <em>Knowledge base management: import files, build indexes (BM25 + Embedding + RRF), search and delete by keyword</em>
</p>

---

## What Makes This Different

**Intent-aware routing.** The same word "Apple" routes to K-line analysis for stock price queries, event impact scoring for acquisition news, or report synthesis for earnings data — decided by the Coordinator agent, not the user.

**Agents don't do work — tools do.** Every agent is a lightweight orchestrator that only calls `self.call_tool()`. All business logic, API calls, and computation live in 32 registered tools across 11 modules. Agents stay thin and testable.

**File upload from browser.** PDF and image files can be uploaded directly via drag-and-drop or file picker in the web UI. Uploaded files are parsed server-side (PyMuPDF for PDF, qwen-vl-plus for images) and routed to both the knowledge base and the LightRAG knowledge graph — no manual directory path needed.

**News auto-imports to KB with quality gates.** Fetched news articles go through a preprocessing pipeline (text cleaning → relevance gate → minimum length filter → doc type classification) before being added to the knowledge base. Duplicates are automatically detected and skipped. Previously, news only stored metadata — now quality articles become retrievable knowledge.

**Collapsible UI with modular frontend.** The web UI uses native ES Modules split into 9 domain-specific JS modules (`modules/`) and 6 layered CSS files (`styles/`). Cards support `data-collapsible` folding for progressive disclosure — users see a clean overview first, expand sections as needed. Score panels are positioned as diagnostic tools below results (collapsed by default), keeping the answer front and center.

**Deep analysis walks the full Agent chain.** News interpretation and topic research go through `orchestrator([AnalysisAgent, ScoringAgent])` — AnalysisAgent does extraction + KB retrieval + LLM analysis, then ScoringAgent evaluates quality and runs hallucination guard. If the agent chain fails, the system falls back to a direct service call. The ScoringAgent is a **universal capability**: any feature can opt in by setting 3 metadata fields (`scoring_source_items`, `scoring_mode`, `scoring_text`).

**Dual-mode smart query.** The query panel offers two modes: "Knowledge Base Q&A" (hybrid BM25+vector retrieval with LLM answer) and "Deep Research" (full 5-phase Pipeline: fetch news → index → AI analysis → structured report → hallucination guard). User-facing labels are translated from technical terms — no "Pipeline", "RAG", or "Agent" jargon visible to users.

**Every chain ends with a quality gate.** The Scoring agent runs a 6-layer hallucination guard (4 rule layers: source grounding, numerical fidelity, citation integrity, structure compliance + 2 LLM layers: LLM critique, LLM assist) and pipeline quality evaluation before any result reaches the user. The guard is **context-aware**: RAG queries expect `[N]` citations and `# Markdown` headers, while deep analysis uses `【】` bracketed sections with relaxed citation requirements — same guard, different scoring criteria. Weight normalization ensures skipped layers don't drag down the overall score.

**Query planning before retrieval.** `QueryPlanner` uses a single LLM call to decompose complex queries (comparisons, timelines, deep dives) into structured sub-queries — each with a source (kb/news/graph/all) and mode (local/global/hybrid/mix). Simple factual queries skip planning and go straight to retrieval.

**Multi-modal document parsing.** PDF files parsed via PyMuPDF (local, no API) and images analyzed via qwen-vl-plus multimodal model. Parsed content from PDFs and images is automatically routed to the knowledge graph for entity-relation extraction.

**Graph RAG — integrated, not just an experiment.** LightRAG knowledge graph is wired into the ingestion pipeline: PDF/image parsed content triggers entity-relation extraction. Agents query the graph on-demand via Function Calling tools (`query_knowledge_graph`, `get_graph_stats`), routed by QueryPlanner's `source: graph` — not forced into every query. Storage is JSON + GraphML files, no external database needed.

**Hybrid retrieval that actually works.** BM25 + ChromaDB ANN vector search (1024-dim) + RRF fusion + qwen3-rerank reranking + metadata filtering + **query expansion** (52 synonym groups + 20 concept association maps, LLM-enhanced for short queries) — six signals combined instead of relying on any single one.

**Domain dictionaries that grow without code changes.** `DictionaryRegistry` centralizes 10 dictionary types (stock mappings, financial terms, synonyms, jieba words, etc.) and auto-merges external JSON files from `data/dictionaries/`. Drop in a JSON file → dictionaries expand at next startup. Coverage stats via `reg.summary()` — weak spots visible at a glance.

**LLM calls never fail silently.** `LLMCaller` wraps every LLM invocation with retry, balanced-bracket JSON parsing, response caching, and anti-hallucination constraints — one unified layer for the entire system.

**609 tests, zero API keys.** All tests pass in under 6 seconds using mocked data sources. LLM and embedding stay real in tests — only external APIs are mocked.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User Layer — FastAPI web UI (dark theme) + CLI             │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Scheduling — AgentRouter (intent → chain)                  │
│               PipelineScheduler (5-phase execution)          │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Query Planning — QueryPlanner (LLM decomposition)          │
│   complex queries → sub-queries with source + mode           │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Agents (4)                                                  │
│  Coordinator  → classifies intent, selects agent chain       │
│  Ingestion    → fetches and indexes source documents         │
│  Analysis     → routes to tools by intent, assembles result  │
│  Scoring      → hallucination guard + quality evaluation     │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌──────────┬─────────────────┬─────────────────────────────────┐
│ Retrieval│ Data APIs       │ LLM Layer                        │
│ BM25+Chroma│ Tushare         │ LLMCaller (retry+cache+JSON)    │
│ RRF+Rerank│ 10jqka/Eastmoney│ ModelRouter (4 tiers, Qwen)    │
│ +Graph   │ LightRAG (integrated) │                                  │
└──────────┴─────────────────┴─────────────────────────────────┘
```

---

## How Routing Works

The Coordinator agent classifies each query into one of five intents, then the scheduler assembles the appropriate agent chain:

| Intent | Example Query | Chain |
|--------|---------------|-------|
| K-line analysis | "Show me the MACD for 600519" | Analysis → Scoring |
| Event impact | "Is this acquisition good or bad?" | Analysis → Scoring |
| Report synthesis | "Summarize Q4 earnings" | Ingestion → Analysis → Scoring |
| News deep-dive | "What happened in AI this week?" | Analysis (deep) → Scoring |
| Topic research | "Research the AI chip sector" | Analysis (deep_topic) → Scoring |
| General | anything else | Ingestion → Analysis → Scoring |

The user types a natural language question. The system decides everything else.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | DashScope Qwen (turbo / plus / max / 235b) |
| Embedding | text-embedding-v3 (1024-dim) |
| Rerank | qwen3-rerank (fallback to RRF score) |
| Backend | FastAPI + 4 async routers + python-multipart (file upload) |
| Frontend | Vanilla JS ES Modules (9 modules) + 6 layered CSS + collapsible cards |
| Data APIs | Tushare, 10jqka, Sina, EastMoney |
| Retrieval | BM25 + ChromaDB (ANN) + RRF + TextChunker + QueryParser (52 synonym groups + 20 concept maps) |
| Query Planning | QueryPlanner (LLM decomposition, 5 intents, source/mode-aware sub-queries) |
| Document Parse | PyMuPDF (PDF, local) + qwen-vl-plus (image multimodal) |
| Graph RAG | LightRAG (integrated: PDF/image → entity-relation extraction → graph query via Function Calling) |
| Domain Dicts | DictionaryRegistry (10 types, JSON-extensible: `data/dictionaries/*.json`) |
| Vector DB | ChromaDB (HNSW ANN, cosine distance, persistent storage) |
| Testing | pytest — 609 tests across 27 files |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-username/financial-rag.git
cd financial-rag

# 2. Create virtual environment
python -m venv myenv

# Activate
# Windows (PowerShell):
.\myenv\Scripts\Activate.ps1
# macOS/Linux:
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and fill in DASHSCOPE_API_KEY
https://github.com/ConardLi
# 5. Launch web UI
python -m financial_rag.main web
# Opens at http://127.0.0.1:8000 — dark theme, sidebar navigation
# Upload PDF/images directly via drag-and-drop in the Data Import panel

# 6. Or use CLI
python -m financial_rag.main pipeline "What happened in AI this week?"
python -m financial_rag.main pipeline "Show me MACD for 600519" -v
python -m financial_rag.main toolcall -l   # list all registered tools
```

---

## Testing

```bash
python -m pytest tests/ -v    # 609 tests, < 6s, no API key needed
```

Tests mock only **data sources** (Tushare, news APIs). LLM, embedding, and rerank stay real. Extraction tools have regex fallback for fully offline operation.

---

## Documentation

- **[Architecture Docs](docs/ARCHITECTURE.md)** — system design, data flow, agent coordination
- **[User Guide (中文)](USER_GUIDE.md)** — feature walkthrough and usage examples
- **[Interview Q&A (中文)](docs/PROJECT_QA.md)** — technical depth for interviews

---

## License

MIT
