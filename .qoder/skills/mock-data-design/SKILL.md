---
name: mock-data-design
description: Correct mock design for Financial RAG — what to mock (data sources only), what NOT to mock (LLM/embedding/rerank), mock data structure requirements, and how mock integrates with the pipeline. Use when adding/modifying mock data, debugging mock mode, or deciding what should be mocked.
---

# Mock Data Design Principles

## Golden Rule: Mock Data Sources Only

```
┌─────────────────────────────────────────────────┐
│  What gets MOCKED          What stays REAL       │
│  ─────────────────         ──────────────        │
│  ✗ Tushare K-line API      ✓ DashScope LLM      │
│  ✗ Tushare stock search    ✓ DashScope Embedding│
│  ✗ Tushare financial data  ✓ DashScope Reranker │
│  ✗ News APIs (10jqka/Sina/EastMoney) ✓ BM25 retrieval │
│                            ✓ Agent processing    │
│                            ✓ Slot filling         │
│                            ✓ Score/evaluation     │
└─────────────────────────────────────────────────┘
```

**Why**: The LLM is the intelligence that processes data. Mocking it makes the entire pipeline meaningless — you'd just get mock responses to mock data. The goal is: **real agents with real intelligence processing realistic mock data**.

## When to Use Mock Mode

| Scenario | MOCK_MODE | Why |
|----------|-----------|-----|
| Local development without Tushare token | `true` | Test data pipeline without API quota |
| Testing news API code | `true` | Domestic APIs may have rate limits |
| Debugging agent logic | `true` | Deterministic data, reproducible results |
| Production / real analysis | `false` | Need real market data + real LLM analysis |
| Testing LLM prompts | `false` | LLM must be real to test prompt quality |

## Mock Data Requirements

### 1. Structure Must Match Real API Exactly

Mock functions return the **exact same types** as real functions:

```python
# tushare_client.py — same signature, same return type
def fetch_stock_kline(ts_code, days, period) -> pd.DataFrame:
    # Real: calls Tushare API
    # Mock: generates realistic DataFrame with OHLCV columns

def search_stock(keyword, limit) -> List[Dict]:
    # Real: calls stock_basic API
    # Mock: returns list of dicts with ts_code, name, market, etc.

# rss_fetcher.py — same return structure
def search_news(keyword, max_news) -> Dict:
    # Returns: {"keyword": str, "total": int, "items": [...], "elapsed_ms": float}
    # Each item: {title, content, source, publish_time, url, sentiment}
```

### 2. Data Must Be Realistic

- K-line prices should follow geometric Brownian motion with realistic volatility
- Stock/ETF codes must use real known codes (600519.SH, 510300.SH, etc.)
- News titles should contain real financial terminology
- Financial indicators should be within plausible ranges for each company type

### 3. Deterministic When Possible

Use seeded random generators so the same request returns consistent data:
```python
np.random.seed(hash(str(base_price) + str(days)) % (2**31))
```

## Implementation Pattern

### Entry-point check (current pattern)

```python
def fetch_stock_kline(ts_code, days=30, period="daily"):
    # Mock check at function entry
    from financial_rag.config import is_mock_enabled
    if is_mock_enabled():
        from financial_rag.mock_data import mock_stock_kline
        return mock_stock_kline(ts_code, days=days, period=period)

    # Real API call...
    api = _get_api()
    ...
```

### Config

```python
# .env
MOCK_MODE=true   # enables mock for Tushare + RSS

# config.py
@dataclass
class MockConfig:
    enable: bool = field(
        default_factory=lambda: os.getenv("MOCK_MODE", "false").lower() == "true"
    )
```

### Web server behavior

When `MOCK_MODE=true`:
- `/api/kline` → mock K-line data + **real LLM analysis**
- `/api/news` → mock news data + **real LLM summary**
- `/api/kb-query` → **real embedding + real rerank** on KB docs
- `/api/pipeline` → **real LLM** processing mock-fetched data

The DASHSCOPE_API_KEY is still required for LLM-dependent endpoints.

## Anti-Patterns (DO NOT)

1. **Never mock the LLM** — it defeats the purpose of having agents
2. **Never mock embeddings** — retrieval quality depends on real vectors
3. **Never mock the reranker** — score calibration depends on real model
4. **Don't return empty/trivial mock data** — agents need enough data to process
5. **Don't use obviously fake values** like "test_stock" or prices of 1.00

## Mock Data Inventory

| Function | Mock | Returns |
|----------|------|---------|
| `fetch_stock_kline()` | `mock_stock_kline()` | DataFrame with 30+ rows of OHLCV |
| `fetch_etf_kline()` | `mock_etf_kline()` | DataFrame with 30+ rows of OHLCV |
| `search_stock()` | `mock_search_stock()` | List of stock dicts from known pool |
| `search_etf()` | `mock_search_etf()` | List of ETF dicts from known pool |
| `fetch_financial_indicators()` | `mock_financial_indicators()` | 4 quarterly reports with metrics |
| `search_news()` | `mock_search_news()` | News search result with keyword-filtered items |
| `fetch_all_news()` | `mock_fetch_all_news()` | Aggregated news from all sources |

All mock functions live in `financial_rag/mock_data.py`.
