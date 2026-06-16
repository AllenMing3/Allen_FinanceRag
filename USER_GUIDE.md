# Financial RAG 使用手册

> 架构概览请看 `README.md`，深入架构细节请看 `docs/ARCHITECTURE.md`。

---

## 1. 环境准备

```cmd
cd d:\llamaindex
myenv\Scripts\activate.bat
```

编辑 `.env`：

| 变量 | 必须 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY=sk-xxx` | 是 | LLM / Embedding / Rerank（不设则走本地模式） |
| `TUSHARE_TOKEN=xxx` | 否 | K线数据（120+ 积分才能用 daily 接口） |
| `MOCK_MODE=true` | 否 | 开启 Mock 模式（详见第 3 节） |

---

## 2. 快速开始

```cmd
python -m financial_rag.main web
```

打开 `http://127.0.0.1:8000`，按顺序操作：

| 步骤 | 页面 | 操作 | 说明 |
|------|------|------|------|
| 1 | 导入数据 | 搜索新闻（如 "AI人工智能"） | 收集元数据，辅助后续文件解析。新闻**不进知识库** |
| 2 | 导入数据 | 分析并导入文件 | Agent 链分析文件 → 抽取指标/实体 → 存入知识库 |
| 3 | 构建知识库 | 构建索引 | BM25 + 向量双通道索引（TextChunker 自动切分） |
| 4 | RAG 查询 | 提问 | AgentRouter **自动路由** → 检索知识库 → LLM 回答（带引用和分数明细） |
| 5 | 智能分析 | 粘贴新闻 / 输入话题 | 抽取指标 + 实体 + KB 上下文 → 利好/利空/中性判断 |
| 6 | 分析工具 | K线分析 | 输入 "茅台" 或 "600519"，生成技术分析报告 |
| 7 | 事件分析 | 事件影响分析 | 输入日期 + 股票，评估事件利好/利空 + 影响因子 |

> **智能路由**：查询自动分类为 5 种意图（kline / event_impact / report / news / general），系统选择最优 Agent 执行链，无需手动指定。

**数据来源：**
- 文件放 `./data/financial` 目录
- 新闻来自国内直连 API（同花顺 / 新浪财经 / 东方财富）
- K线来自 Tushare Pro（需 120+ 积分）

---

## 3. Mock 模式

无需 Tushare Token，K线和新闻使用内置模拟数据，LLM 仍走真实 API。

`.env` 中加 `MOCK_MODE=true`，或命令行启动：

```cmd
set MOCK_MODE=true && python -m financial_rag.main web
```

| 数据源 | Mock 效果 |
|--------|----------|
| 新闻 | 25 条内置 AI 行业新闻 |
| K线 | 8 只股票 + 7 只 ETF 的模拟行情（几何布朗运动） |
| 话题调研 | 使用 mock 新闻数据，不调用真实 API |
| LLM / Embedding | **真实 DashScope API**（需 Key） |

Web UI 开启 Mock 模式时会显示橙色提示条。

---

## 4. 测试

```cmd
:: 全量（322 tests，无需 API Key）
python -m pytest tests/ -v
```

按需运行单个模块：

| 模块 | 命令 | 覆盖内容 |
|------|------|----------|
| Agent 路由 | `pytest tests/test_agent_router.py -v` | 意图分类、链选择、元数据提取 |
| Agent | `pytest tests/test_new_agents.py -v` | Coordinator / Analysis / Scoring |
| 新工具 | `pytest tests/test_new_tools.py -v` | scoring / coordinator / report / event_impact |
| 工厂配线 | `pytest tests/test_factory.py -v` | 4 Agent 注册、链顺序、工具绑定 |
| 数据合并 | `pytest tests/test_orchestrator_merge.py -v` | metadata merge、findings extend |
| 原始 Agent | `pytest tests/test_agents.py -v` | Ingestion + Analysis + 完整链 |
| 智能分析 | `pytest tests/test_analysis.py -v` | 新闻分析 + 话题调研 (mock) |
| 抽取工具 | `pytest tests/test_extraction_tools.py -v` | 5 个抽取工具 + 长文章 |
| 分析工具 | `pytest tests/test_analysis_tools.py -v` | 增长率、比率、对比、汇总 |
| Mock 数据 | `pytest tests/test_mock_data.py -v` | K线、搜索、指标、新闻 |
| LLM 调用层 | `pytest tests/test_llm_caller.py -v` | LLMCaller 重试、JSON、缓存、约束 |
| 数据编排器 | `pytest tests/test_data_orchestrator.py -v` | 多池摄入/搜索/跨池检索 |
| 查询解析器 | `pytest tests/test_query_parser.py -v` | 意图、实体、日期提取 |

> 测试只 mock 数据源，LLM / Embedding / Rerank 保持真实。抽取工具走 regex fallback，无需 API Key。

---

## 5. 命令速查表

| 命令 | 用途 | 示例 |
|------|------|------|
| `web` | Web UI | `python -m financial_rag.main web` |
| `pipeline` | 端到端分析（自动路由） | `python -m financial_rag.main pipeline "商汤营收" -v` |
| `pipeline` | K线分析（自动识别） | `python -m financial_rag.main pipeline "茅台走势" -v` |
| `pipeline` | 事件分析（自动识别） | `python -m financial_rag.main pipeline "2024-06-01 发生了什么" -v` |
| `news` | 拉新闻 | `python -m financial_rag.main news "AI" -s` |
| `kline` | K线分析 | `python -m financial_rag.main kline "商汤" --days 30` |
| `toolcall` | Function Calling | `python -m financial_rag.main toolcall "商汤营收" -v` |
| `slot` | 槽位填充 | `python -m financial_rag.main slot "商汤营收" -t fin` |
| `score` | 检索打分 | `python -m financial_rag.main score "商汤算力" -k 5` |
| `build` | 构建知识库 | `python -m financial_rag.main build --dir ./data/financial` |
| `query` | 交互查询 | `python -m financial_rag.main query -i` |

Pipeline 模板选项：`-t quick`（默认）/ `-t fin`（财报）/ `-t news`（新闻）/ `-t deep`（深度）

---

## 6. Agent 体系

| Agent | 职责 | 触发方式 |
|-------|------|----------|
| **CoordinatorAgent** | 意图分类 + Agent 链选择 | 手动调起（`call_tool(classify_query_intent)`） |
| **IngestionAgent** | 数据摄取 + 元数据提取 | report / news / general 链 |
| **AnalysisAgent** | 统一分析引擎（指标抽取 + K线分析 + 事件影响 + 报告生成） | 所有链，通过 `intent` 元数据选择工具链 |
| **ScoringAgent** | 全链路评分 + 防幻觉校验 | **所有链末端** |

**AnalysisAgent 意图路由：**

| intent | 工具链 |
|--------|-------|
| `kline` | fetch_kline_report → analyze_kline → generate_kline_analysis |
| `event_impact` | fetch_date_events → assess_event_impact |
| `general` | extract_financial_metrics → extract_entities → synthesize_report |

**核心设计原则：**
- Agent 只做编排决策（`call_tool()`），不包含任何业务逻辑
- 所有业务能力实现在 tools 层
- 每条链都以 ScoringAgent 结尾，确保输出质量
- 所有 LLM 调用经过 LLMCaller 保护层（重试 + JSON 解析 + 缓存 + 防幻觉约束）

---

## 7. 常见问题

**Rerank 403** — `qwen3-rerank` 需在 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 手动开通，未开通时自动降级为 RRF 融合。

**新闻获取不到** — 关键词太偏或 API 频率限制，换热门主题重试。

**中文乱码** — CMD 下执行 `set PYTHONUTF8=1`。

**K线获取失败** — 检查 `.env` 中有 `TUSHARE_TOKEN` 且积分 >= 120。

**详细日志** — 大部分命令加 `-v` 可输出详细过程。
