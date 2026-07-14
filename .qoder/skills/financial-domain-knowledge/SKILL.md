---
name: financial-domain-knowledge
description: Chinese financial market data domain knowledge — stock/ETF codes, K-line OHLCV data, technical indicators (MACD/RSI/KDJ/Bollinger), financial metrics (EPS/ROE/margins), Tushare API conventions. Use when working with K-line data, technical analysis, stock/ETF code handling, or financial indicator computation.
---

# Financial Domain Knowledge

## Chinese Stock/ETF Code Conventions

### Stock Codes

| Exchange | Prefix | Example | Suffix |
|----------|--------|---------|--------|
| Shanghai Main Board | 600xxx, 601xxx, 603xxx | 600519.SH (贵州茅台) | .SH |
| Shenzhen Main Board | 000xxx, 001xxx | 000858.SZ (五粮液) | .SZ |
| Shenzhen SME Board | 002xxx | 002594.SZ (比亚迪) | .SZ |
| ChiNext (创业板) | 300xxx | 300750.SZ (宁德时代) | .SZ |
| STAR Market (科创板) | 688xxx | 688981.SH (中芯国际) | .SH |

### ETF Codes

| Exchange | Prefix | Example |
|----------|--------|---------|
| Shanghai | **51xxxx**.SH | 510300.SH (沪深300ETF), 510050.SH (上证50ETF) |
| Shenzhen | **159xxx**.SZ | 159915.SZ (创业板ETF), 159995.SZ (芯片ETF) |

**Critical**: ETF detection must use `startswith("51")` or `startswith("159")`, NOT `startswith("5")` or `startswith("15")` — the latter matches non-ETF codes.

## K-Line Data (OHLCV)

Standard columns returned by Tushare and mock:

```
date       open     high     low      close    volume    amount
2024-01-02 1680.00  1695.50  1675.00  1690.00  3500000   5915000000
```

- **OHLC**: Open/High/Low/Close prices for the period
- **Volume**: Number of shares traded
- **Amount**: Total transaction value (volume × price)
- Data is sorted by date ascending, typically daily or weekly frequency

## Technical Indicators

### Moving Averages (MA)
```
MA5  = mean(last 5 closes)   — short-term trend
MA10 = mean(last 10 closes)  — medium-term trend
MA20 = mean(last 20 closes)  — medium-long trend
```
Golden cross: short MA crosses above long MA → bullish signal

### MACD (Moving Average Convergence Divergence)
```
EMA12 = exponential moving average, span=12
EMA26 = exponential moving average, span=26
DIF   = EMA12 - EMA26
DEA   = EMA(DIF, span=9)
MACD bar = (DIF - DEA) × 2

Signals:
- 金叉 (golden cross): DIF crosses above DEA → bullish
- 死叉 (death cross): DIF crosses below DEA → bearish
- 多头 (bullish): DIF > DEA (sustained)
- 空头 (bearish): DIF < DEA (sustained)
```

### RSI (Relative Strength Index, 14-day)
```
RSI = 100 - 100/(1 + avg_gain/avg_loss)

Signals:
- RSI > 70: 超买 (overbought)
- RSI < 30: 超卖 (oversold)
- 30-70: 中性 (neutral)
```

### Bollinger Bands (20-day)
```
Middle = MA20
Upper  = MA20 + 2 × StdDev20
Lower  = MA20 - 2 × StdDev20

Position signals:
- Price near upper band: potential resistance
- Price near lower band: potential support
```

### KDJ
```
RSV = (close - low9) / (high9 - low9) × 100
K = EMA(RSV, com=2)
D = EMA(K, com=2)
J = 3K - 2D

Signals:
- K crosses above D: 金叉 (golden cross, bullish)
- K crosses below D: 死叉 (death cross, bearish)
```

## Financial Indicators (from Tushare `fina_indicator`)

| Metric | Chinese | Meaning |
|--------|---------|---------|
| EPS | 每股收益 | Earnings per share |
| ROE | 净资产收益率 | Return on equity |
| grossprofit_margin | 毛利率 | Gross profit margin |
| netprofit_margin | 净利率 | Net profit margin |
| debt_to_assets | 资产负债率 | Debt-to-asset ratio |
| current_ratio | 流动比率 | Current assets / current liabilities |
| ocfps | 每股经营现金流 | Operating cash flow per share |
| bps | 每股净资产 | Book value per share |

## Tushare Pro API

- Base URL: `https://api.tushare.pro`
- Auth: token in header
- Rate limits: vary by endpoint; `daily` requires 120+ points (积分)
- Common endpoints used:
  - `daily()` — stock daily K-line
  - `weekly()` — stock weekly K-line
  - `fund_daily()` — ETF daily K-line
  - `fina_indicator()` — financial indicators
  - `stock_basic()` — stock list (cache with TTL!)
  - `fund_basic()` — ETF list (cache with TTL!)

### Caching Strategy
`stock_basic()` and `fund_basic()` return full lists — cache with 1-hour TTL to avoid repeated full-list fetches on every search call.

## News Data Sources (Domestic Direct APIs)

| Source | Type | API Endpoint |
|--------|------|--------------|
| 东方财富搜索 | Keyword search | `search-api-web.eastmoney.com/search/jsonp` (JSON param) |
| 同花顺 | Real-time news | `news.10jqka.com.cn/tapp/news/push/stock/` |
| 新浪财经 | General finance | `feed.mix.sina.com.cn/api/roll/get` (pageid=153, lid=2516) |

**Note**: All sources use direct domestic HTTP APIs via httpx. feedparser is available as a fallback for custom RSS feeds.
