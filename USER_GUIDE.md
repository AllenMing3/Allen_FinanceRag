# FinRAG 使用手册

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

打开 `http://127.0.0.1:8000`，通过顶部 4 个标签页操作：

| 标签页 | 功能 | 操作 | 说明 |
|--------|------|------|------|
| **系统概览** | 架构展示 | 查看 4 Agent 协作链 | Coordinator → Ingestion → Analysis → Scoring，意图路由示例 |
| **数据管理** | 导入数据 | 搜索新闻（如 "AI人工智能"） | 收集元数据，辅助后续文件解析。新闻**不进知识库** |
| **数据管理** | 导入数据 | 分析并导入文件 | Agent 链分析文件 → 抽取指标/实体 → 存入知识库（后台 LLM 分析 + 进度显示） |
| **数据管理** | 构建索引 | 点击「构建索引」 | BM25 + ChromaDB 向量双通道索引（jieba trigram 分词 + TextChunker 自动切分，Chroma 持久化到 `data/knowledge_base/chroma/`） |
| **数据管理** | 管理知识库 | 按来源/关键词搜索删除 | 查看各来源文档数，一键删除匹配关键词的文档 |
| **智能查询** | RAG 查询 | 输入问题 | QueryParser **查询扩展**（同义词 + 概念关联）→ QueryPlanner **LLM 拆解**（复杂查询分解为子查询）→ AgentRouter 自动路由 → 检索知识库 → LLM 回答（带引用和分数明细） |
| **智能查询** | K线分析 | 输入股票代码或名称 | 生成 MACD/RSI/KDJ/布林带技术分析报告 |
| **深度分析** | 新闻解读 | 粘贴新闻文本 | 多维影响评估 + 关键信号 + 风险提示 |
| **深度分析** | 话题调研 | 输入话题关键词 | 子话题聚类 + 情绪趋势 + 反向信号 |
| **深度分析** | 追问 | 分析完成后输入追问 | 基于原始分析上下文的多轮对话，支持对新闻/话题结果深入追问 |

> **智能路由**：查询自动分类为 5 种意图（kline / event_impact / report / news / general），系统选择最优 Agent 执行链，无需手动指定。
>
> **查询规划**：复杂查询（对比、时间线、深度分析）先经 `QueryPlanner` 拆解为多个子查询，每个子查询带有来源（kb/news/graph/all）和模式（local/global/hybrid/mix），简单查询直接检索。

**数据来源：**
- 文件放 `./data/financial` 目录
- 新闻来自国内直连 API（同花顺 / 新浪财经 / 东方财富）
- K线来自 Tushare Pro（需 120+ 积分）

---

## 2.5 知识库管理

在「数据管理」标签页可以进行数据导入和知识库维护：

| 功能 | 说明 |
|------|------|
| **新闻抓取** | 输入关键词（如"AI"）+ 条数（10/20/30/50），拉取国内新闻作为元数据 |
| **文件勾选** | 每个文件带 checkbox，支持全选/取消 |
| **内容预览** | 点击文件名预览前 20 行 |
| **分析模式** | 🔍 深度分析 (LLM 抽取指标+实体) / ⚡ 快速导入 (跳过 LLM) |
| **关键词过滤** | 只拉取匹配关键词的新闻 |
| **自定义目录** | 直接输入路径导入非默认目录的文件 |

知识库维护：

| 功能 | 说明 |
|------|------|
| **来源查看** | 显示各来源的文档数量，如 `news: 73`, `nvidia_2025: 1` |
| **关键词搜索** | 输入关键词（如"商汤"）搜索匹配的文档 |
| **关键词删除** | 删除所有匹配关键词的文档 |
| **清空知识库** | 一键清空全部文档 + 重置进度 |

> 文件导入后 LLM 分析在后台运行，页面会显示进度百分比。分析完成后结论自动存入知识库。

---

## 2.6 持续学习

每次新闻解读或话题调研的分析结论会自动存入知识库，供未来分析参考：

- **自动存储**：分析完成后，结论以 `analysis_result` 类型写入 KB，去重更新
- **学习历史面板**：在「深度分析」页面底部查看历史学习记录（时间、话题、研判结论）
- **KB 来源标识**：学习记录显示为 `analysis:news:xxx` 或 `analysis:topic:xxx`，可按来源管理

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
:: 全量（521 tests，无需 API Key）
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
| Smoke | `pytest tests/test_smoke.py -v` | Web API 全端点、文件预览、KB 去重 |
| 持久化 | `pytest tests/test_persistence.py -v` | 索引保存/加载、去重、备份轮转 |
| 抽取工具 | `pytest tests/test_extraction_tools.py -v` | 5 个抽取工具 + 长文章 |
| 分析工具 | `pytest tests/test_analysis_tools.py -v` | 增长率、比率、对比、汇总 |
| Mock 数据 | `pytest tests/test_mock_data.py -v` | K线、搜索、指标、新闻 |
| LLM 调用层 | `pytest tests/test_llm_caller.py -v` | LLMCaller 重试、JSON、缓存、约束 |
| 数据编排器 | `pytest tests/test_data_orchestrator.py -v` | 多池摄入/搜索/跨池检索 |
| 查询解析器 | `pytest tests/test_query_parser.py -v` | 意图、实体、日期提取、查询扩展（同义词 + 概念关联） |
| K线工具 | `pytest tests/test_kline_tools.py -v` | K线获取、技术分析 |
| 新闻工具 | `pytest tests/test_news_tools.py -v` | 股票新闻、财经新闻、公告 |
| 事件影响工具 | `pytest tests/test_event_impact_tools.py -v` | 事件获取、影响评估 |
| 深度分析工具 | `pytest tests/test_deep_analysis_tools.py -v` | 新闻深度、话题深度 |
| 协调器工具 | `pytest tests/test_coordinator_tools.py -v` | 意图分类、链选择 |
| 报告工具 | `pytest tests/test_report_tools.py -v` | 报告合成 |
| Tushare 计算 | `pytest tests/test_tushare_compute.py -v` | K线统计、技术指标、analyze_kline |

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
| **AnalysisAgent** | 统一分析引擎（指标抽取 + K线分析 + 事件影响 + 报告生成 + 深度新闻/话题分析） | 所有链，通过 `intent` 元数据选择工具链 |
| **ScoringAgent** | 全链路评分 + 防幻觉校验 | **所有链末端** |

**AnalysisAgent 意图路由：**

| intent | 工具链 |
|--------|-------|
| `kline` | fetch_kline_report → analyze_kline → generate_kline_analysis |
| `event_impact` | fetch_date_events → assess_event_impact |
| `news` | analyze_news_deep (多维影响 + 关键信号 + 风险 + 后续关注) |
| `general` | extract_financial_metrics → extract_entities → synthesize_report |

**核心设计原则：**
- Agent 只做编排决策（`call_tool()`），不包含任何业务逻辑
- 所有业务能力实现在 tools 层
- 每条链都以 ScoringAgent 结尾，确保输出质量
- 所有 LLM 调用经过 LLMCaller 保护层（重试 + 平衡括号 JSON 解析 + 缓存 + 防幻觉约束）
- 防幻觉校验为 6 层透明检查：L1 来源锚定（jieba 分词重叠）、L2 数值一致、L3 引用完整、L4 结构规范、L5 LLM 质疑（LLM 审查答案 + 来源，输出结构化发现）、L6 LLM 协助（低分时 LLM 修复问题）
- `QueryPlanner` 在 QueryParser 之后、检索之前：用一次 LLM call 将复杂查询拆解为子查询，简单查询跳过规划直接检索

---

## 6.5 查询规划 (QueryPlanner)

复杂查询（对比分析、时间线梳理、深度调研）在检索之前先经过 `QueryPlanner` 拆解：

| 查询类型 | 示例 | 子查询数 | 策略 |
|----------|------|---------|------|
| factual | "矛台收盘价多少" | 1 | 直接检索 |
| comparison | "英伟达 vs 华为芯片" | 3 | 并行 |
| timeline | "OpenAI 融资历程" | 3 | 顺序 |
| deep_dive | "商汤生成式AI前景" | 4 | 并行 |
| summary | "AI行业最近怎么样" | 2-3 | 并行 |

**每个子查询包含**：
- `query`: 子查询文本
- `source`: 数据来源 (kb / news / graph / all)
- `mode`: 检索模式 (local / global / hybrid / mix)
- `purpose`: 查询目的

**设计原则**：
- 一次 LLM call，JSON 输出，解耦现有 QueryParser / Retriever
- LLM 调用失败时自动降级为单子查询（全来源 + mix 模式）
- `is_simple=True`（单子查询）时跳过规划开销

---

## 7. 字典扩展

系统的检索质量依赖领域字典（股票映射、金融术语、同义词、jieba 分词词表等）。`DictionaryRegistry` 统一管理 10 种字典，支持**不改代码**扩展：

**扩展方式**：把 JSON 文件放入 `data/dictionaries/` 目录，启动时自动加载并合并：

```
data/dictionaries/
├── ai_domain.json        # AI/科技领域术语
└── stocks_extended.json  # 股票映射扩展
```

**当前覆盖情况**（内置 + 外部 JSON 合并后）：

| 字典 | 规模 | 说明 |
|------|------|------|
| stock_map | 33 条 | 股票名称/别名 → 代码映射 |
| financial_terms | 64 个 | BM25 高权重词 |
| industry_terms | 73 个 | BM25 中权重词 |
| synonym_lookup | 143 条 (52 组) | 同义词双向扩展 |
| concept_map | 20 组 | 概念关联（如“芯片”→半导体、光刻等） |
| jieba_words | 91 个 | jieba 分词扩展 |

**补强方法**：发现某个领域检索不准时，检查对应字典是否覆盖。例如缺少某公司别名，在 `stocks_extended.json` 的 `stock_map` 中加一行即可。

**查看当前状态**：
```python
from financial_rag.retrievers.dictionary_registry import get_registry
print(get_registry().summary())  # 一眼看清哪里弱
```

---

## 8.5 LightRAG 图谱实验

`experiments/lightrag_experiment.py` 是一个独立 PoC，用 LightRAG SDK 从中文财经新闻中抽取知识图谱：

**运行**：
```cmd
python experiments/lightrag_experiment.py
```

**结果**（5 篇 AI 新闻）：
- 60 个实体、59 条关系
- 10 种实体类型（organization / data / artifact / concept / person / event 等）
- 导出 GraphML 格式，可用 Gephi 可视化

**4 种查询模式**：

| 模式 | 适用场景 | 示例 |
|------|---------|------|
| `local` | 实体属性查询 | "商汤科技的营收" |
| `global` | 跨文档全局模式 | "AI行业发展趋势" |
| `hybrid` | 多跳实体关系 | "OpenAI 和商汤的关系" |
| `mix` | 多路融合 | "中国AI政策受益者" |

**注意**：这是独立实验，尚未集成到主系统。运行前需安装 `pip install lightrag-hku numpy`。

---

## 9. 常见问题

**Rerank 403** — `qwen3-rerank` 需在 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 手动开通，未开通时自动降级为 RRF 融合。

**新闻获取不到** — 关键词太偏或 API 频率限制，换热门主题重试。

**中文乱码** — CMD 下执行 `set PYTHONUTF8=1`。

**K线获取失败** — 检查 `.env` 中有 `TUSHARE_TOKEN` 且积分 >= 120。

**详细日志** — 大部分命令加 `-v` 可输出详细过程。

**服务端启动慢** — 首次启动会在后台线程初始化知识库和组件（`_ensure_init()`），服务立即接受请求，若初始化未完成则端点会等待。

**知识库删除后重建索引慢** — 已优化：按关键词/来源删除时使用 `HybridRetriever.remove()` 过滤文档 + 同步删除 ChromaDB 中对应向量，仅重建 BM25，无需全量重建 embedding。
