# Financial RAG 使用手册

> 面向使用者的操作指南。架构细节请看 `README.md`。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据流总览](#2-数据流总览)
3. [Web UI 操作指南](#3-web-ui-操作指南)
4. [Pipeline — 一句话出分析报告](#4-pipeline--一句话出分析报告)
5. [CLI 工具集](#5-cli-工具集)
6. [常见问题排查](#6-常见问题排查)

---

## 1. 环境准备

```powershell
cd d:\llamaindex
.\myenv\Scripts\Activate.ps1
```

激活成功后终端前面会有 `(myenv)` 标志。

**验证环境：**

```powershell
python -c "import tushare, httpx; print('tushare:', tushare.__version__, 'httpx:', httpx.__version__)"
python -c "from financial_rag.llm import get_llm; print('llm ok')"
```

**API Key：** 编辑 `.env` 文件，填入：
- `DASHSCOPE_API_KEY=sk-你的密钥` — LLM/Embedding/Rerank。没有 Key 也能跑，降级到纯本地模式。
- `TUSHARE_TOKEN=你的token` — K线数据接口。在 [tushare.pro](https://tushare.pro) 注册获取，需要 120+ 积分才能调用日线接口。

---

## 2. 数据流总览

这个系统的核心价值是：**分析文件建立知识 → 收集新闻提供上下文 → 基于知识库回答问题**。

### 2.1 三类数据源及各自的角色

```
📁 导入文件（财报/研报/文本）
  → 真正的知识原料
  → 经过 Agent 分析（抽取指标、实体）→ 进入知识库

📰 新闻搜索（国内直连 API：同花顺/新浪财经/东方财富）
  → 只是元数据（时间、关键词、来源）
  → 不进知识库，但提供：解析先验 + 查询上下文

📈 K 线数据（Tushare Pro：股票/ETF 行情）
  → 按需查询，不进知识库
  → KLineAgent 分析技术指标 + LLM 生成解读
```

**简单规则：**
- 知识库的内容只来自**经过分析的文件**
- 新闻元数据有双重角色：**解析先验**（导入时注入 Agent prompt）+ **查询上下文**
- 只有 `text` 字段会被 BM25 + Embedding 向量化，其他信息（来源、时间、关键词）都是 metadata

### 2.2 Web UI 路径：数据从哪来、变成什么、到哪去

**第 1 步：搜索新闻 → 收集元数据（先验知识）**

先搜新闻，收集元数据。这些元数据不只是查询时的补充，还是文件解析时的先验知识。

```
用户搜索 "AI人工智能"
       ↓
国内直连 API: 同花顺 + 新浪财经 + 东方财富
       ↓
存入 news_metadata.json  ← 元数据（标题、关键词、时间、来源）
存入 news_archive.jsonl  ← 原始数据存档（追加模式）
生成 output/*.md          ← 可读报告
```

元数据的结构：

```json
{
  "keyword": "AI、人工智能",
  "title": "字节跳动收购行歌科技",
  "source": "同花顺",
  "publish_time": "2025-06-07 14:30:00",
  "fetched_at": "2025-06-07 15:00:00"
}
```

**关键：新闻不进 KB，不向量化。它的作用有两个：**
1. **解析先验** — 导入文件时，元数据注入 Agent 的 LLM prompt，辅助识别公司、主题、事件
2. **查询上下文** — 查询时匹配关键词，作为补充信息显示

**第 2 步：导入文件 → Agent 分析（带元数据上下文）→ 知识库**

点"分析并导入"，系统读取目录下的文件，对每篇文档运行 Agent 链。
此时 Agent 已经拿到了新闻元数据作为背景知识，解析更精准：

```
新闻元数据（先验知识）
       ↓ 注入 LLM prompt
原始文件 (.jsonl / .txt / .json)
       ↓
IngestionAgent  →  清洗文本 + 提取元数据（来源、公司、日期、文档类型）
       ↓
ExtractionAgent →  抽取财务指标（营收/利润/毛利率/EPS...）+ 实体（公司/事件）
       ↓
KB 文档（原文 + 富化元数据）→ kb_docs.json
```

每篇 KB 文档的结构：

```json
{
  "text": "贵州茅台2024年营收1738亿元，净利润862亿元...",
  "meta": {
    "source": "上交所公告",
    "analyzed": true,
    "metrics": {"revenue": "...", "net_income": "...", "eps": "..."},
    "entities": [{"company": "贵州茅台", "event_type": "..."}]
  }
}
```

**写到哪？** `data/knowledge_base/kb_docs.json`（JSON 数组，每次导入后保存，启动时加载）。

**第 3 步：构建索引**

点“构建索引”，系统对 `kb_docs.json`（分析后的文件）建立 BM25 + Embedding 索引：

```
                    ┌──→ BM25 索引（关键词倒排表）
每篇文档的 text ────┤
                    └──→ Embedding 向量（text-embedding-v3, 1024维）
```

| 索引类型 | 输入 | 产出 | 存在哪 |
|---------|------|------|--------|
| BM25 | text 分词后的关键词 | 倒排索引 | **内存**（不持久化） |
| Embedding | text 原文 | 1024维向量 | **内存**（不持久化） |

重启后自动从 `kb_docs.json` 重建。

**第 4 步：RAG 查询**

```
用户问题
       ↓
  ┌────┴────┐
  ↓         ↓
BM25 检索   向量检索（语义匹配）
  └────┬────┘
       ↓
   RRF 融合 → Top-K 篇知识库文档
       ↓
   + 相关新闻元数据（从 news_metadata.json 匹配）
       ↓
   LLM 生成回答（标注引用）
```

查询结果包含两部分：
1. **知识库来源** — 检索到的分析文档（带分数、来源、抽取的指标）
2. **相关新闻动态** — 匹配的新闻元数据（标题、时间、来源）

**第 5 步：K线技术分析（可选）**

在「工具」页面，输入股票名称或代码，实时获取 K 线数据并生成技术分析：

```
用户输入 "茅台" 或 "600519"
       ↓
KLineAgent 识别股票代码 (600519.SH)
       ↓
Tushare Pro 拉取日K/周K 数据
       ↓
计算技术指标: MA / MACD / RSI / 布林带 / KDJ
       ↓
LLM 生成自然语言技术分析
       ↓
返回: 统计数据 + 指标信号 + AI 分析解读
```

特点：
- 按需查询，不存入知识库
- 支持股票 + ETF（自动识别，如 600519.SH / 510050.SH / 159995.SZ）
- 可选择日 K 或周 K，支持 30/60/120 天回溯
- 输出: 统计数据 + MA/MACD/RSI/布林带/KDJ 信号 + AI 分析解读

### 2.3 什么被向量化？什么只是 metadata？

| 数据类型 | 被向量化的内容 | 只作为 metadata（不向量化） |
|---------|--------------|------------------------|
| 导入文件 | .txt 全文 / .jsonl 的 text 字段 | source, file path, metrics, entities |
| K 线查询 | **不向量化** | 按需查询，技术指标 + LLM 分析结果直接返回 |
| Pipeline 获取 | `title\ncontent` 拼成的文本 | source, publish_time, url |
| 新闻搜索 | **不向量化** | 仅存为元数据（keyword, title, time） |

### 2.4 所有数据存储位置

| 文件 | 格式 | 内容 | 什么时候写入 |
|------|------|------|------------|
| `data/knowledge_base/kb_docs.json` | JSON 数组 | 经分析后的知识文档 | 每次文件导入后保存 |
| `data/knowledge_base/news_metadata.json` | JSON 数组 | 新闻元数据（标题/关键词/时间） | 每次搜新闻后保存 |
| `data/knowledge_base/news_archive.jsonl` | JSONL（追加） | 新闻原始数据存档 | 每次搜新闻后追加 |
| `output/*.md` | Markdown | 新闻汇总报告、分析报告 | 搜新闻/pipeline 时生成 |

---

## 3. Web UI 操作指南

```powershell
python -m financial_rag.main web
# 打开 http://127.0.0.1:8000
```

### 典型工作流程

**第 1 步：搜新闻（收集元数据 / 先验知识）**

在「数据源」页面，搜新闻（如"AI人工智能"）。新闻**不进知识库**，只存为元数据：
- `data/knowledge_base/news_metadata.json`（标题、关键词、时间）
- `data/knowledge_base/news_archive.jsonl`（原始数据存档）
- `output/*.md`（可读报告）

搜多个主题，元数据会累加。这些元数据有两个用途：
- **解析先验** — 导入文件时注入 Agent，辅助识别公司/主题/事件
- **查询上下文** — 查询时自动匹配相关新闻作为补充信息

**第 2 步：导入文件（知识来源）**

点目录旁的"分析并导入"按钮。系统会：
1. 读取目录下的文件（.jsonl / .txt / .json）
2. 对每篇文档运行 IngestionAgent + ExtractionAgent（**带元数据上下文**）
3. 抽取财务指标（营收、利润、EPS...）和实体（公司、事件）
4. 存入 `kb_docs.json`

可以把自定义文件放到 `./data/financial` 目录。

**第 3 步：构建索引**

切到「构建知识库」，点“构建索引”。对知识库文档建立 BM25 + 1024维向量索引。

**第 4 步：提问**

切到「RAG 查询」，输入问题。系统展示：
- 检索到的知识库文档（带分数、来源、抽取的指标）
- 相关新闻动态（元数据匹配的实时新闻）
- LLM 生成的回答（标注引用来源）

**第 5 步：K线分析（可选）**

切到「工具」，输入股票名称或代码（如 "茅台"、"600519"、"人工智能ETF"）。
系统自动识别股票/ETF，实时拉取行情数据，计算技术指标，LLM 生成技术分析解读。
数据源: Tushare Pro（需在 `.env` 中配置 `TUSHARE_TOKEN`，积分 ≥ 120）。

---

## 4. Pipeline — 一句话出分析报告

```powershell
# 基本用法
python -m financial_rag.main pipeline "茅台2024年利润增长情况"

# 新闻简报模板 + 详细日志
python -m financial_rag.main pipeline "新能源板块最近有什么利好" -t news -v

# 深度分析 + 输出到文件
python -m financial_rag.main pipeline "降准对银行股的影响" -t deep -o ./output
```

**4 种模板：**

| 参数 | 场景 |
|------|------|
| `-t quick` | 快速回答（默认） |
| `-t fin` | 财报数据分析 |
| `-t news` | 新闻摘要 |
| `-t deep` | 多维度深度分析 |

加 `-v` 可以看到每个 Agent 的执行详情和耗时。

---

## 5. CLI 工具集

### 新闻抓取

```powershell
python -m financial_rag.main news "AI人工智能" -s
```

- 从东方财富拉新闻 → 保存 `output/*.md`
- 追加到 `data/knowledge_base/news_archive.jsonl`（原始存档）
- 新闻只作为元数据，不进知识库
- 加 `-s` 生成 AI 摘要

### ETF / 股票 K 线

```powershell
python -m financial_rag.main kline "人工智能ETF" --days 30 -s
python -m financial_rag.main kline "茅台" --days 60 -s
```

支持股票和 ETF，自动识别代码。数据源: Tushare Pro（需 `.env` 中配置 `TUSHARE_TOKEN`）。

### Function Calling

```powershell
python -m financial_rag.main toolcall "茅台营收增长多少" -v
```

LLM 自动选择工具获取数据。

### 槽位填充

```powershell
python -m financial_rag.main slot "茅台2024年利润" -t financial_report
```

用模板约束 LLM 输出格式。

### 检索打分

```powershell
python -m financial_rag.main score "茅台营收增长" -k 5
```

不调 LLM，纯测检索链路质量。

### 知识库构建

```powershell
python -m financial_rag.main build --dir ./data/financial
```

### 交互查询

```powershell
python -m financial_rag.main query -i
```

### 命令速查表

| 命令 | 用途 | 示例 |
|------|------|------|
| `web` | Web UI | `python -m financial_rag.main web` |
| `pipeline` | 端到端分析 | `python -m financial_rag.main pipeline "查询" -v` |
| `news` | 拉新闻 | `python -m financial_rag.main news "AI" -s` |
| `kline` | 股票/ETF K线 | `python -m financial_rag.main kline "茅台" --days 30` |
| `toolcall` | Function Calling | `python -m financial_rag.main toolcall "查询" -v` |
| `slot` | 槽位填充 | `python -m financial_rag.main slot "查询" -t financial_report` |
| `score` | 检索打分 | `python -m financial_rag.main score "查询" -k 5` |
| `build` | 构建知识库 | `python -m financial_rag.main build --dir ./data` |
| `query` | 交互查询 | `python -m financial_rag.main query -i` |
| `analyze` | Multi-Agent 分析 | `python -m financial_rag.main analyze 文件.pdf` |
| `demo` | 演示 | `python -m financial_rag.main demo` |

---

## 6. 常见问题排查

### "Rerank 403 Access Denied"

`gte-rerank` 需要在[阿里云百炼控制台](https://bailian.console.aliyun.com/)手动开通。不开通不影响使用，自动降级到 RRF 融合排序。

### "新闻获取不到数据"

- 关键词太偏 → 换个热门主题试试
- API 访问受限 → 同花顺 / 新浪财经 / 东方财富均有访问频率限制，可稍后重试
- 网络问题 → 确认能访问 `news.10jqka.com.cn`、`feed.mix.sina.com.cn`、`search-api-web.eastmoney.com`

### "中文乱码"

用 PowerShell 而不是 CMD。或 `set PYTHONUTF8=1`。

### "想看详细日志"

大部分命令加 `-v`：

```powershell
python -m financial_rag.main pipeline "查询" -v
python -m financial_rag.main toolcall "查询" -v
```

### "LLM 回答有幻觉"

1. 降低温度：`config.py` 里 `temperature` 改为 `0.0`
2. 用槽位填充（`slot` 命令）代替自由生成
3. 查看打分卡（`score` 命令）定位哪个环节弱
