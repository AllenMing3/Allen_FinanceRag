# Financial RAG — LLM-Powered RAG Analysis Pipeline

End-to-end retrieval-augmented generation pipeline with Multi-Agent orchestration, hybrid search, and self-scoring feedback loop. Built on Alibaba DashScope (Qwen).

```
User Query → Data Fetch → RAG Index → Multi-Agent Process → Slot-Filled Output → Score & Evolve
```

---

## Architecture Overview

Three core engines power the system. All are domain-agnostic and defined by abstract interfaces.

```
┌─────────────────────────────────────────────────────────────┐
│                   PipelineScheduler (5-phase)                │
│     Fetch → Index → Process → Output → Evolve               │
├─────────────┬───────────────────┬────────────────────────────┤
│  Coordinate │     Indexer       │        Reflection          │
│             │                   │                            │
│ AgentOrch-  │ PipelineOrch-     │ ReflectionLoop             │
│ estrator    │ estrator          │  Think→Act→Observe→Judge   │
│             │                   │                            │
│ SEQUENTIAL  │ BM25 + Vector     │ HallucinationGuard         │
│ PARALLEL    │ + RRF Fusion      │  L1-L6 six-layer check     │
│ CONDITIONAL │ + gte-rerank      │                            │
└─────────────┴───────────────────┴────────────────────────────┘
```

| Engine | File | Responsibility |
|--------|------|----------------|
| **Coordinate** | `core/orchestrator.py` | Register agents, decide execution order, pass context between them. Supports 3 modes: sequential, parallel, conditional. |
| **Indexer** | `core/indexer.py` | 4-stage retrieval pipeline: Clean → Extract → Retrieve → Verify. Hybrid BM25+Vector with RRF fusion. |
| **Reflection** | `core/reflector.py` | ReAct reasoning loop (Think→Act→Observe→Judge) + 6-layer anti-hallucination guard. |

---

## 5-Phase Pipeline

```
Phase 1: Fetch    Function Calling auto-selects data tools (akshare / MCP)
Phase 2: Index    Documents → BM25 + Vector → RRF fusion → gte-rerank → Top-K
Phase 3: Process  Multi-Agent chain: Ingestion → Extraction → Analysis → Forecast → Report
Phase 4: Output   Slot Filling with template formatting (4 templates available)
Phase 5: Evolve   PipelineScoreCard scoring + HallucinationGuard verification
```

### Agent Chain (Phase 3 detail)

```
IngestionAgent   → Clean text + extract metadata (source/company/date/doc_type)
ExtractionAgent  → Pull financial metrics (revenue/profit/EPS/ROE) + entities
AnalysisAgent    → 5-dimension analysis (profitability/growth/health/efficiency/valuation)
ForecastAgent    → 3-scenario projection (optimistic/baseline/pessimistic)
ReportAgent      → 3-format output (summary/detailed/PPT outline)
```

### Retrieval Modes

| Mode | Condition | Chain |
|------|-----------|-------|
| Local only | No API Key | BM25 + Jaccard → RRF |
| With Embedding | API Key set | BM25 + Vector → RRF |
| Full pipeline | API Key active | BM25 + Vector → RRF → gte-rerank |

### Model Routing

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
| `config.py` → `financial_rag/config.py` | All settings: LLM, RAG, Coordinator, Pipeline, Reflection, MCP |
| `.env.example` | Env var template (`DASHSCOPE_API_KEY`) |
| `requirements.txt` | pip dependencies |

### `financial_rag/` — Core Package

| File | Role | Key exports |
|------|------|-------------|
| `__init__.py` | Package entry, version `2.0.0` | Re-exports all public APIs |
| `main.py` | CLI entry (argparse) | `main()` |
| `config.py` | Global config dataclasses | `config`, `AppConfig`, `LLMConfig`, `RAGConfig`, `MCPConfig` |
| `prompts.py` | LLM prompt templates + few-shot examples | — |
| `templates.py` | 4 slot templates: QUICK_QA, FINANCIAL_REPORT, NEWS_BRIEF, DEEP_ANALYSIS | `SlottedTemplate`, `ALL_TEMPLATES` |
| `slot_filler.py` | Parallel slot filling engine with TTFT measurement | `SlotFiller`, `create_slot_filler` |
| `news_fetcher.py` | Real-time news via akshare (stock/financial/announcements) | `fetch_stock_news`, `fetch_financial_news` |
| `etf_fetcher.py` | ETF market data via akshare | `search_etf`, `fetch_etf_kline` |

### `financial_rag/core/` — Architecture Layer

| File | Role | Key exports |
|------|------|-------------|
| `base.py` | Abstract foundations | `BaseAgent`, `AgentContext`, `AgentResult`, `ExecutionMode` |
| `orchestrator.py` | Multi-Agent scheduling engine | `AgentOrchestrator`, `CoordinatorConfig` |
| `pipeline.py` | 5-phase PipelineScheduler (central dispatcher) | `PipelineScheduler`, `PipelineResult` |
| `router.py` | CLI command dispatch + all command handlers | `CommandRouter` |
| `factory.py` | Factory functions for component creation | `create_orchestrator`, `create_hybrid_retriever`, `setup_environment` |
| `indexer.py` | Hybrid retrieval pipeline (Clean→Extract→Retrieve→Verify) | `PipelineOrchestrator`, `PipelineConfig` |
| `reflector.py` | ReAct loop + 6-layer HallucinationGuard | `ReflectionLoop`, `HallucinationGuard` |
| `scorer.py` | Full-pipeline scorecard (per-stage scoring + diagnosis) | `PipelineScoreCard`, `ScoreGrade` |
| `protocol.py` | Agent messaging: AgentMessage + MessageBus + MessageAdapter | `AgentMessage`, `MessageBus` |

### `financial_rag/agents/` — 5 Specialized Agents

| File | Role |
|------|------|
| `ingestion_agent.py` | Data ingestion: load files/text → clean + auto-extract metadata |
| `extraction_agent.py` | Feature extraction: financial metrics + entities + search queries |
| `analysis_agent.py` | Multi-dimensional analysis: profitability/growth/health/efficiency/valuation |
| `forecast_agent.py` | Trend forecasting: 3 scenarios (optimistic/baseline/pessimistic) |
| `report_agent.py` | Report generation: summary / detailed / PPT outline |

### `financial_rag/llm/` — LLM Layer

| File | Role |
|------|------|
| `dashscope_client.py` | DashScope API wrapper: LLM + Embedding + Rerank |
| `model_router.py` | Auto-select model by task complexity + budget control |

### `financial_rag/tools/` — Function Calling Registry

| File | Role |
|------|------|
| `core.py` | Infrastructure: `FunctionDef`, `FunctionRegistry`, `ToolExecutor`, `ToolCallSession` |
| `news_tools.py` | Registered tool: fetch news → save as Markdown report |
| `kline_tools.py` | Registered tool: fetch ETF K-line → save as analysis report |

### `financial_rag/mcp_client/` — MCP Integration

| File | Role |
|------|------|
| `client.py` | Generic MCP client (stdio transport) |
| `news_client.py` | china-stock-mcp news tool wrapper |

### `financial_rag/retrievers/`

| File | Role |
|------|------|
| `__init__.py` | `HybridRetriever`: BM25 + Vector + RRF fusion + Rerank |

### `financial_rag/data/`

| File | Role |
|------|------|
| `financial_news.jsonl` | Sample training data (3 news articles) |

---

## CLI Quick Reference

All commands run from project root `d:\llamaindex` with venv activated.

```bash
# Unified Pipeline (recommended)
python -m financial_rag.main pipeline "茅台2024年利润增长情况"
python -m financial_rag.main pipeline "新能源板块利好" -t news -v
python -m financial_rag.main pipeline "降准对银行股的影响" -t deep -o ./output

# Interactive query
python -m financial_rag.main query -i
python -m financial_rag.main query -q "茅台毛利率"

# Build knowledge base
python -m financial_rag.main build --dir ./data/financial

# Multi-Agent analysis
python -m financial_rag.main analyze ./report.pdf --parallel

# Function Calling
python -m financial_rag.main toolcall -l                    # list all tools
python -m financial_rag.main toolcall "茅台营收增长" -v      # single call
python -m financial_rag.main toolcall "对比分析" --multi-turn # multi-turn

# Slot filling
python -m financial_rag.main slot "茅台财报" -t financial_report

# News fetching
python -m financial_rag.main news "AI新闻" -s               # + LLM summary

# ETF K-line
python -m financial_rag.main kline "人工智能ETF" --days 30 -s

# Retrieval scoring
python -m financial_rag.main score "茅台营收" -k 5
python -m financial_rag.main score "汇率走势" --json scores.json

# Demo
python -m financial_rag.main demo
```

Template options: `quick` (default) | `fin` | `news` | `deep`

---

## Configuration

All config in `financial_rag/config.py`. Global instance: `from financial_rag.config import config`.

| Section | Key params | Defaults |
|---------|-----------|----------|
| `config.llm` | model, embedding_model, rerank_model, temperature, max_tokens | qwen-plus, text-embedding-v3, gte-rerank, 0.0, 4096 |
| `config.coordinator` | execution_mode, max_parallel_agents, max_retries, timeout_seconds | sequential, 3, 2, 300 |
| `config.pipeline` | hybrid_top_k, rrf_k, bm25_weight, vector_weight, min_faithfulness | 10, 60, 0.3, 0.7, 0.7 |
| `config.rag` | vector_store_path, similarity_top_k, chunk_size, chunk_overlap | ./storage/..., 5, 512, 50 |
| `config.reflection` | max_retrievals, max_steps, min_confidence, hallucination_threshold | 3, 6, 0.6, 0.6 |
| `config.mcp` | enable_mcp, china_stock_mcp_dir, timeout | false, "", 30.0 |

Env vars (`.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `DASHSCOPE_API_KEY` | No | Alibaba DashScope API key. Without it, falls back to local-only mode. |

---

## Programming API

```python
# LLM call
from financial_rag.llm import get_llm
from financial_rag.config import config
llm = get_llm(api_key=config.llm.api_key, model="qwen-plus")
resp = llm.chat("分析茅台2024年毛利率")

# Hybrid retrieval
from financial_rag.retrievers import HybridRetriever
from financial_rag.llm import get_embedding, get_reranker
retriever = HybridRetriever(
    embedder=get_embedding(api_key=config.llm.api_key),
    reranker=get_reranker(api_key=config.llm.api_key),
)
retriever.index(docs)
results = retriever.search("茅台盈利", top_k=5)

# Multi-Agent orchestration
from financial_rag.core import create_orchestrator
orch = create_orchestrator()
result = orch.execute("./data/financial/report.pdf")

# Function Calling
from financial_rag.tools import create_financial_registry, create_tool_session
registry = create_financial_registry()
session = create_tool_session(llm=get_llm(), registry=registry)
stats = session.run("茅台营收增长多少")

# Model routing
from financial_rag.llm import ModelRouter
router = ModelRouter()
router.override("analysis_agent", "qwen-max")
```

---

## Debug Guide

| Problem | Where to look |
|---------|---------------|
| Agent not working | `core/orchestrator.py` `_apply_updates()` — check context passing. Set `config.coordinator.verbose = True`. |
| News fetch fails | Test: `from financial_rag.news_fetcher import fetch_stock_news; fetch_stock_news("600519", max_news=3)`. akshare uses East Money API. |
| Retrieval inaccurate | Run `python -m financial_rag.main score "query"`. Adjust `config.pipeline` weights. |
| LLM hallucination | Check `core/reflector.py` HallucinationGuard 6-layer scores. Lower `temperature` to 0. Raise `min_faithfulness`. |
| Model cost too high | Check `llm/model_router.py` BudgetConfig. Override non-critical agents to `qwen-turbo`. |
| Adding a new Agent | Inherit `BaseAgent` from `core/base.py`, implement `process()`, register in `core/factory.py:create_orchestrator()`. |

---

## Setup (Windows)

```powershell
cd d:\llamaindex
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env, fill in DASHSCOPE_API_KEY

# Verify
python -c "from financial_rag.news_fetcher import HAS_AKSHARE; print('akshare:', HAS_AKSHARE)"
python -c "from financial_rag.llm import get_llm; print('llm ok')"
```

| Issue | Fix |
|-------|-----|
| `UnicodeDecodeError: 'gbk'` | `set PYTHONUTF8=1` or use PowerShell |
| `Activate.ps1` blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `lxml` build fails | `pip install akshare --only-binary :all:` |
| Chinese output garbled | Use PowerShell instead of CMD |

---

## License

MIT
