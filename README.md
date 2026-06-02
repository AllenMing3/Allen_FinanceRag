# Financial RAG — 财报/经济新闻智能分析系统

基于阿里百炼 DashScope (Qwen) + MCP 数据源的金融端到端分析 Pipeline。

## 核心流水线

```
用户一句话 → MCP 实时获取数据 → RAG 索引与检索 → 加工分析 → 槽位填充输出 → 评分反馈自进化
```

不再是一个个独立的命令行工具，而是从**获取**到**筛选**到**加工**到**输出**到**进化**的连贯处理链路。

## 模块索引

出问题了去哪个文件修，一目了然。

```
financial_rag/
├── main.py                     # CLI 入口：统一 Pipeline 入口 + 工厂函数
├── config.py                   # 全局配置：LLM/RAG/Coordinator/Reflection 参数
├── tools.py                    # Function Calling 能力注册中心 + ToolExecutor
├── news_fetcher.py             # 实时新闻获取（MCP china-stock-mcp） + akshare 降级
├── etf_fetcher.py              # ETF 行情数据获取（MCP 优先）
├── prompts.py                  # LLM prompt 模板 + Few-shot 示例
├── templates.py                # 槽位模板定义（4 种：快答/财报/新闻/深分）
├── slot_filler.py              # 槽位填充引擎（并行填充 + TTFT 测量）

├── llm/                        # LLM 层
│   ├── dashscope_client.py     #   阿里百炼客户端：LLM + Embedding + Rerank
│   └── model_router.py         #   模型路由：按任务复杂度自动选模型+预算控制

├── agents/                     # 5 个 Agent（Multi-Agent 流水线）
│   ├── ingestion_agent.py      #   摄取：读文件/文本 → 提取 metadata
│   ├── extraction_agent.py     #   抽取：财务指标 + 实体 + 检索查询
│   ├── analysis_agent.py       #   分析：5 维度分析
│   ├── forecast_agent.py       #   预测：3 种情景推演
│   └── report_agent.py         #   报告：3 种格式输出

├── core/                       # 三大核心架构
│   ├── coordinator.py          #   Coordinate：Agent 编排+调度+MessageBus
│   ├── protocol.py             #   AgentMessage 消息协议 + 数据链路追溯
│   ├── indexer.py              #   Indexer：混合检索流水线 (BM25+Vector+RRF+Rerank)
│   ├── reflector.py            #   Reflection：六层防幻觉校验
│   └── scorer.py               #   PipelineScoreCard：全链路打分卡

├── retrievers/
│   └── __init__.py             #   HybridRetriever：混合检索器实现

├── ingestion/
│   └── __init__.py             #   文档加载器（JSONL/TXT/PDF）

├── middleware/
│   └── __init__.py             #   中间件（日志/缓存/错误处理）

└── data/
    └── financial_news.jsonl    #   练手样本数据（3 篇财经新闻）
```

## 依赖清单

每个包干什么用，出问题知道怀疑谁。

| 包 | 用途 | 哪里用到 |
|---|---|---|
| `dashscope` | 阿里百炼 SDK，调 Qwen 系列模型 | `llm/dashscope_client.py` |
| `akshare` | A 股数据源（MCP 不可用时的降级方案） | `news_fetcher.py`, `etf_fetcher.py` |
| `llama-index` | RAG 框架：文档加载、切分、向量存储 | `core/indexer.py` |
| `jieba` | 中文分词，BM25 检索的前置 | `retrievers/__init__.py` |
| `pandas` | DataFrame 处理，新闻数据清洗 | `news_fetcher.py`, `ingestion_agent.py` |
| `pydantic` | 数据模型校验 (AgentMessage 等) | `core/protocol.py`, `tools.py` |
| `python-dotenv` | 从 `.env` 文件加载 API Key | `config.py` |
| `python-magic` | 文件类型检测（MIME） | `ingestion/__init__.py` |
| `regex` | 高级正则，metadata 自动提取 | `ingestion_agent.py` |
| `typing-extensions` | 类型注解补丁 (Python 3.10 兼容) | 全局 |

## MCP 集成 — 数据获取层

数据获取已统一通过 **china-stock-mcp**（CodeBuddy MCP 扩展）完成，不再直接依赖 akshare。

### MCP 工具能力

MCP 配置文件：`c:\Users\<用户名>\.codebuddy\mcp.json`

| 工具名 | 功能 | Pipeline 环节 | 状态 |
|---|---|---|---|
| `get_hist_data` | 个股/指数历史 K 线 | 获取 | ✅ 已验证 |
| `get_stock_research_report` | 个股研报 | 获取 | ✅ 已验证 |
| `get_balance_sheet` | 资产负债表 | 获取 | 待验证 |
| `get_income_statement` | 利润表 | 获取 | 待验证 |
| `get_cash_flow` | 现金流量表 | 获取 | 待验证 |
| `get_financial_metrics` | 财务指标汇总 | 获取+分析 | 待验证 |
| `get_stock_a_code_name` | 股票代码→名称 | 获取 | 待验证 |
| `get_stock_basic_info` | 股票基本信息 | 获取 | 待验证 |
| `get_fund_flow` | 资金流向 | 分析 | 待验证 |
| `get_inner_trade_data` | 内部交易 | 分析 | 待验证 |
| `get_macro_data` | 宏观经济数据 | 分析 | 待验证 |
| `get_investor_sentiment` | 市场情绪 | 分析 | 待验证 |
| `get_stock_value` | 估值数据 | 分析 | 待验证 |
| `get_stock_volatility` | 波动率 | 分析 | 待验证 |
| `get_stock_technical_rank` | 技术面排名 | 分析 | 待验证 |
| `get_stock_cyq` | 筹码分布 | 分析 | 待验证 |
| `get_news_data` | 财经新闻（服务端 bug，暂不可用） | 获取 | ❌ 已知问题 |
| `get_realtime_data` | 实时行情（代理拦截） | 获取 | ❌ 已知问题 |

### akshare 降级

当 MCP 工具不可用或返回异常时，`news_fetcher.py` 和 `etf_fetcher.py` 自动降级到 akshare（本地安装）。接口设计为 MCP 优先，降级对上层透明。

### 新闻能力清单（`news_fetcher.py`）

| 函数 | 功能 | MCP 工具对应 |
|---|---|---|
| `fetch_stock_news("600519")` | 个股近期新闻 | `get_news_data`（故障降级 akshare） |
| `fetch_financial_news(keyword="降准")` | 关键词搜索/最新快讯 | `get_news_data`（故障降级 akshare） |
| `fetch_announcements("600519")` | 公司公告 | `get_news_data`（故障降级 akshare） |

### 扩展 MCP 工具

如果要加新的数据能力，流程：

1. 在 CodeBuddy 的 MCP 扩展中安装/添加对应 MCP Server
2. 在 `financial_rag/tools.py` 的 `FunctionRegistry` 中注册对应的代理函数
3. Agent / Pipeline 层即可自动发现和调用，无需改上层代码

---

## 环境准备

### 前置要求

- Python 3.10+
- 推荐用 **PowerShell** 而不是 CMD（UTF-8 编码，不会 GBK 报错）

### Windows 开发环境

#### 1. 克隆 + 创建虚拟环境

```powershell
# PowerShell（推荐）
cd d:\llamaindex
python -m venv myenv
.\myenv\Scripts\Activate.ps1
# 如果报"无法加载文件"，先执行：
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

```cmd
:: CMD（备用）
cd d:\llamaindex
python -m venv myenv
myenv\Scripts\activate
```

> 激活成功会有 `(myenv)` 前缀。所有后续操作都在虚拟环境里做。

#### 2. 安装依赖

```powershell
# PowerShell（直接装）
pip install -r requirements.txt
```

```cmd
:: CMD 需要加 PYTHONUTF8，否则 requirements.txt 里的中文注释会报 GBK 解码错误
set PYTHONUTF8=1
pip install -r requirements.txt
```

> 也可以一劳永逸：`setx PYTHONUTF8 1` 设成用户环境变量，重开终端生效。

#### 3. 配置 API Key

```powershell
# PowerShell
Copy-Item .env.example .env
notepad .env
```

`.env` 文件内容：

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> Key 获取：https://bailian.console.aliyun.com/
>
> 没有 Key 也能跑 — 系统自动降级到纯本地模式（BM25 + Jaccard）。

#### 4. 验证安装

```powershell
python -c "from financial_rag.news_fetcher import HAS_AKSHARE; print('akshare:', HAS_AKSHARE)"
python -c "from financial_rag.llm import get_llm; print('llm ok')"
```

看到 `akshare: True` + `llm ok` 就是环境好了。

#### 5. Windows 踩坑清单

| 现象 | 原因 | 解决 |
|---|---|---|
| `UnicodeDecodeError: 'gbk' codec can't decode...` | CMD 用 GBK 读 UTF-8 文件 | `set PYTHONUTF8=1` 或换 PowerShell |
| `python` 找不到 | Python 没加 PATH | 安装时勾选 "Add Python to PATH" |
| `Activate.ps1` 禁止执行 | PowerShell 执行策略 | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` 报 permission | 不在虚拟环境里 | 确认有 `(myenv)` 前缀 |
| 中文输出乱码 | CMD 默认 GBK | 换 PowerShell，或在脚本头加 `sys.stdout.reconfigure(encoding='utf-8')` |
| `lxml` 编译失败 | 部分 Windows 缺 VS Build Tools | `pip install akshare --only-binary :all:` |

### Linux 生产部署（后期迁移参考）

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 Key，或 export DASHSCOPE_API_KEY=sk-xxx
```

---

## CLI 命令速查

所有命令都从项目根目录 `d:\llamaindex` 执行，且虚拟环境已激活。

### 🆕 统一 Pipeline（推荐，计划中）

> **目标入口**：一条命令跑完整链路，代替下面分散的子命令。

```bash
# 端到端：MCP获取数据 → RAG索引 → 分析 → 输出 → 打分
python -m financial_rag.main pipeline "茅台2024年利润增长情况"
python -m financial_rag.main pipeline "新能源板块最近有什么利好" --output report.md
python -m financial_rag.main pipeline "帮我分析一下最近要降准对银行股的影响" --verbose
```

### 当前子命令（逐步迁入 Pipeline）

#### 演示 & 查询

```bash
# 演示模式（跑一条内置示例）
python -m financial_rag.main demo

# 交互式查询（可以连续问问题）
python -m financial_rag.main query -i

# 单次查询
python -m financial_rag.main query -q "茅台2024年营收增长了多少？"
```

#### 知识库

```bash
# 构建知识库（指定目录，支持 JSONL/TXT/PDF）
python -m financial_rag.main build --dir ./data/financial

# 检索打分测试（不调 LLM，只测检索质量）
python -m financial_rag.main score "茅台营收增长" -k 5
python -m financial_rag.main score "汇率走势" --json scores.json
```

#### Multi-Agent 分析

```bash
# 串行执行（默认）
python -m financial_rag.main analyze ./data/financial/maotai_2024.pdf

# 并行执行
python -m financial_rag.main analyze ./data/financial/maotai_2024.pdf --parallel
```

#### Function Calling / 工具调用

```bash
# 列出所有已注册能力（包括 MCP 工具）
python -m financial_rag.main toolcall -l

# Function Calling 单次调用
python -m financial_rag.main toolcall "茅台营收增长多少" -v

# 强制必须调工具
python -m financial_rag.main toolcall "计算茅台毛利率" --tool-choice required

# 多轮调用（LLM 可以多次选工具）
python -m financial_rag.main toolcall "茅台和五粮液利润对比" --multi-turn -v
```

#### 槽位填充

```bash
# 交互模式，支持切换模板（输入 fin/quick/news/deep）
python -m financial_rag.main query -i

# 对比测试：自由生成 vs 槽位填充
python -m financial_rag.main slot "茅台2024年利润增长情况" -t financial_report

# 仅槽位填充（不跑对照组）
python -m financial_rag.main slot "茅台营收" -t quick_qa --no-freeform
```

---

## 配置参考

所有配置项在 `financial_rag/config.py`。全局实例：`from financial_rag.config import config`。

### 环境变量（`.env`）

| 变量 | 说明 | 必填 | 默认值 |
|---|---|---|---|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key | 否 | 无 Key 降级本地模式 |

### LLM 配置 (`config.llm`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `api_key` | 从 `.env` 读取 | 阿里百炼 Key |
| `model` | `qwen-plus` | 对话模型：turbo/plus/max/qwen3-235b |
| `embedding_model` | `text-embedding-v3` | 向量模型，1024 维 |
| `rerank_model` | `gte-rerank` | 重排序模型 |
| `temperature` | `0.0` | 生成温度（0=确定性，1=随机） |
| `max_tokens` | `4096` | 单次最大输出 token |

### 模型分层与路由 (`llm/model_router.py`)

| 层级 | 模型 | 适用任务 |
|---|---|---|
| LIGHT | qwen-turbo | 简单解析、格式化 |
| STANDARD | qwen-plus | 抽取、分类 |
| HEAVY | qwen-max | 多步推理、趋势分析 |
| ULTRA | qwen3-235b | 综合分析、报告生成 |

修改 Agent 使用的模型：

```python
from financial_rag.llm import ModelRouter

router = ModelRouter()
router.override("analysis_agent", "qwen-max")     # 分析 Agent 用大模型
router.override("ingestion_agent", "qwen-turbo")  # 摄取 Agent 用小模型
```

### Coordinator 配置 (`config.coordinator`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `execution_mode` | `sequential` | sequential / parallel / conditional |
| `max_parallel_agents` | 3 | 并行时最多同时跑几个 Agent |
| `max_retries` | 2 | Agent 失败重试次数 |
| `timeout_seconds` | 300 | 单个 Agent 超时 |
| `verbose` | True | 是否打印详细日志 |

### RAG 配置 (`config.rag`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `vector_store_path` | `./storage/financial_vector_store` | 向量存储路径 |
| `similarity_top_k` | 5 | 检索返回数量 |
| `chunk_size` | 512 | 文档切分大小 |
| `chunk_overlap` | 50 | 切分重叠量 |

### Pipeline 配置 (`config.pipeline`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hybrid_top_k` | 10 | 混合检索粗排数量 |
| `rrf_k` | 60 | RRF 融合参数 |
| `bm25_weight` | 0.3 | BM25 权重 |
| `vector_weight` | 0.7 | 向量检索权重 |
| `min_faithfulness` | 0.7 | 防幻觉最低忠实度 |

---

## 数据流 & 架构

### 统一 Pipeline（目标架构）

```
用户一句话
  ↓
┌─ 阶段1: 获取 ─────────────────────────────────────────────┐
│  MCP (china-stock-mcp) 实时拉取                              │
│  ├─ get_hist_data / get_income_statement / ...               │
│  └─ 降级: akshare (MCP 不可用时)                             │
├─ 阶段2: 索引 ─────────────────────────────────────────────┤
│  RAG 混合检索 → BM25 + Vector → RRF → gte-rerank              │
├─ 阶段3: 加工 ─────────────────────────────────────────────┤
│  Function Calling 自动选工具 → Multi-Agent 串行/并行分析       │
│  Ingestion → Extraction → Analysis → Forecast → Report        │
├─ 阶段4: 输出 ─────────────────────────────────────────────┤
│  槽位填充 (Slot Filling) → 多种模板格式化                     │
│  FINANCIAL_REPORT / NEWS_BRIEF / QUICK_QA / DEEP_ANALYSIS    │
├─ 阶段5: 进化 ─────────────────────────────────────────────┤
│  PipelineScoreCard → 六层防幻觉 → 反馈调优                    │
└────────────────────────────────────────────────────────────────┘
```

### Agent 流水线（Multi-Agent 模式，阶段3的子步骤）

```
输入(文本/PDF/JSONL)
  ↓
IngestionAgent     → 摄取：清洗文本 + 自动提取 metadata（source/company/date/doc_type...）
  ↓
ExtractionAgent    → 抽取：财务指标（营收/净利/毛利率/EPS/ROE...）+ 实体（公司/人物/事件...）
  ↓
AnalysisAgent      → 分析：5 维度（盈利能力/成长性/财务健康/运营效率/估值）
  ↓
ForecastAgent      → 预测：3 情景（乐观/基准/悲观）
  ↓
ReportAgent        → 报告：3 格式（摘要/详细/PPT 大纲）
```

### 检索流水线

```
用户查询
  ↓
Jieba 分词
  ↓
BM25 关键词检索 ─┐
                  ├→ RRF 融合 → gte-rerank 精排 → Top-K 结果
向量语义检索 ─────┘
```

### 三条检索模式自动切换

| 模式 | 条件 | 链路 |
|---|---|---|
| 纯本地 | 无 API Key | BM25 + Jaccard → RRF |
| 带 Embedding | 有 API Key | BM25 + Vector → RRF |
| 全链路 | API Key 可用 | BM25 + Vector → RRF → gte-rerank |

---

## 调试指南

### "为什么某个 Agent 不工作？"

1. 查 `core/coordinator.py` 的 `_apply_updates()` — 上游 Agent 写了什么 context_updates
2. 开启 `config.coordinator.verbose = True` 看每个 Agent 的输入输出
3. 如果启用了 MessageBus：`orchestrator.get_data_lineage()` 追踪完整链路

### "新闻拉不下来？"

1. **先确认 MCP 是否可用**：检查 `c:\Users\<用户名>\.codebuddy\mcp.json` 中 `china-stock-mcp` 已配置
2. MCP 工具 `get_news_data` 已知有服务端 bug → 系统会自动降级到 akshare
3. 验证 akshare 降级链路是否正常：
   ```python
   from financial_rag.news_fetcher import fetch_stock_news
   r = fetch_stock_news("600519", max_news=3)
   print(r["total"], r["elapsed_ms"])
   ```
4. 如果返回 0 条但没报错：该股票当天确实没新新闻，换只热门股试试
5. 如果报网络错误：检查网络，akshare 走东方财富接口
6. 如果需要代理：akshare 支持 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量

### "检索结果不准？"

1. 跑 `python -m financial_rag.main score "你的查询"` 看各阶段打分
2. 分低的是瓶颈 — 比如 BM25 低 → 加更多同义词到文档；Rerank 低 → 检查文档质量
3. 调整 `config.pipeline` 的权重参数（bm25_weight / vector_weight）
4. 试试不同的 `similarity_top_k` 值

### "LLM 输出幻觉？"

1. 检查 `core/reflector.py` 的 HallucinationGuard 六层评分
2. 降低 `config.llm.temperature`（设成 0）
3. 提高 `config.pipeline.min_faithfulness` 阈值
4. 开启 slot_filler（槽位填充比自由生成更不容易胡编）

### "模型调用太贵？"

1. 检查 `llm/model_router.py` 的 BudgetConfig
2. 手动 override 非关键 Agent 用 turbo：
   ```python
   router.override("ingestion_agent", "qwen-turbo")
   router.override("extraction_agent", "qwen-turbo")
   ```
3. 查看 `router.stats` 统计各模型调用次数和成本

### "想加一个新 Agent？"

1. 继承 `agents/` 下的 `BaseAgent`（参考 `analysis_agent.py`）
2. 实现 `process(self, context: AgentContext) -> AgentResult`
3. 在 `main.py` 的 `create_orchestrator()` 里注册
4. Agent 之间通过 `AgentContext` 的 dict 字段或 MessageBus 传数据

---

## 编程接口速查

### 基础 LLM 调用

```python
from financial_rag.llm import get_llm
from financial_rag.config import config

llm = get_llm(api_key=config.llm.api_key, model="qwen-plus")
resp = llm.chat("茅台2024年毛利率是多少？")
print(resp.content)
```

### 模型路由器

```python
from financial_rag.llm import ModelRouter

router = ModelRouter()
# 自动路由（按 Agent 名推断复杂度）
llm = router.get_llm_for_agent("analysis_agent")
# 手动覆盖
router.override("report_agent", "qwen-max")
# 查看统计
print(router.stats.total_calls, router.stats.total_cost)
```

### 混合检索

```python
from financial_rag.retrievers import HybridRetriever
from financial_rag.llm import get_embedding, get_reranker

retriever = HybridRetriever(
    embedder=get_embedding(api_key=config.llm.api_key),
    reranker=get_reranker(api_key=config.llm.api_key),
)
retriever.index(docs)
results = retriever.search("茅台盈利", top_k=5, use_rerank=True)
```

### Multi-Agent

```python
from financial_rag.main import create_orchestrator

orch = create_orchestrator()
result = orch.execute("./data/financial/report.pdf")
# 每个 Agent 的结果
for r in result.agent_results:
    print(f"[{'OK' if r.success else 'FAIL'}] {r.agent_name}: {r.message}")
```

### Function Calling

```python
from financial_rag.tools import create_financial_registry, create_tool_session
from financial_rag.llm import get_llm

registry = create_financial_registry()
session = create_tool_session(llm=get_llm(), registry=registry)
stats = session.run("茅台营收增长多少")
print(stats.final_answer)
```

### MCP 数据获取（推荐）

数据获取首选 MCP，`news_fetcher.py` 已将 MCP 包装为 Function Calling 工具，LLM 自动调用。

```python
from financial_rag.news_fetcher import fetch_stock_news, fetch_financial_news, fetch_announcements

# 个股新闻（MCP 优先，失败自动降级 akshare）
r = fetch_stock_news("600519", max_news=5)
for item in r["items"]:
    print(item["title"], item["publish_time"])

# 关键词搜索
r = fetch_financial_news(keyword="降准", max_news=10)

# 公司公告
r = fetch_announcements("600519", max_news=20)
```

---

## 项目结构（完整）

```
llamaindex/
├── .env.example                  # 环境变量模板（可提交）
├── .gitignore                    # Git 忽略规则
├── README.md                     # 本文件
├── requirements.txt              # pip 依赖
├── main.py                       # 根入口（可选）
│
└── financial_rag/                # 核心包
    ├── __init__.py               #   包导出
    ├── main.py                   #   CLI 入口 + 工厂函数
    ├── config.py                 #   全局配置
    ├── tools.py                  #   Function Calling 能力注册
    ├── news_fetcher.py           #   实时新闻（MCP 优先 + akshare 降级）
    ├── etf_fetcher.py            #   ETF 行情（MCP 优先）
    ├── prompts.py                #   LLM prompt 模板
    ├── templates.py              #   槽位模板
    ├── slot_filler.py            #   槽位填充引擎
    │
    ├── agents/
    │   ├── ingestion_agent.py    #   摄取 Agent
    │   ├── extraction_agent.py   #   抽取 Agent
    │   ├── analysis_agent.py     #   分析 Agent
    │   ├── forecast_agent.py     #   预测 Agent
    │   └── report_agent.py       #   报告 Agent
    │
    ├── core/
    │   ├── coordinator.py        #   Agent 编排调度 + MessageBus
    │   ├── protocol.py           #   AgentMessage 消息协议
    │   ├── indexer.py            #   混合检索流水线
    │   ├── reflector.py          #   六层防幻觉
    │   └── scorer.py             #   全链路打分卡
    │
    ├── llm/
    │   ├── dashscope_client.py   #   阿里百炼客户端
    │   └── model_router.py       #   模型路由 + 预算控制
    │
    ├── retrievers/
    │   └── __init__.py           #   HybridRetriever
    │
    ├── ingestion/
    │   └── __init__.py           #   文档加载器
    │
    ├── middleware/
    │   └── __init__.py           #   中间件
    │
    └── data/
        └── financial_news.jsonl  #   练手数据（3 篇新闻）
```

---

## 许可证

MIT License
