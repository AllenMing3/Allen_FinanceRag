# FinRAG 系统架构（中文版）

> 面向开发者和用户的系统理解指南。以用户操作路径为主线，说明每个模块的目标、边界和质量基线。
> 英文详细版见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 1. 系统全景

```
用户操作                    系统内部
────────                    ────────
导入数据 ──→ 预处理流水线 ──→ 知识库(KB) + 知识图谱(LightRAG)
智能查询 ──→ 5阶段Pipeline ──→ Agent链 ──→ 防幻觉校验 ──→ 返回结果
深度分析 ──→ Agent链(Analysis+Scoring) ──→ 抽取+检索+LLM ──→ 防幻觉 ──→ 返回结果
K线分析  ──→ Tushare实时数据 ──→ 技术指标计算 ──→ 返回报告
```

---

## 2. 端到端数据流

### 2.1 导入路径

```
新闻抓取 ──→ 预处理（清洗→相关性门控→长度过滤→分类）──→ 去重 ──→ kb_docs + news_metadata
文件导入 ──→ Agent链分析（IA→EA→Scoring）──→ 预处理 ──→ kb_docs + LightRAG图谱
文件上传 ──→ PDF解析(PyMuPDF)/图片解析(qwen-vl-plus) ──→ 预处理 ──→ kb_docs + LightRAG图谱
```

### 2.2 查询路径（5 阶段 Pipeline）

```
POST /api/pipeline → PipelineScheduler.run(query)

Phase 1 Fetch  — Function Calling 自动选数据源（news_tools / kline_tools）
                  → 标准化文档列表 fetched_data
Phase 2 Index  — TextPreprocessor 清洗 → Retriever.add() 增量索引
                  → Retriever.search()（内部: QueryParser扩展 → BM25+Vector → RRF → Rerank → Filter）
                  → retrieved_items
Phase 3 Process — AgentRouter.route()（4 种意图: kline/event_impact/report/news + general fallback）
                   → 动态选 Agent 链 → orchestrator.execute()
Phase 4 Output — SlotFiller 槽位填充（Agent 已有输出时跳过）
Phase 5 Evolve — PipelineScoreCard 全链路打分 + HallucinationGuard(mode="rag") 防幻觉
```

> QueryParser 在 Retriever.search() 内部调用，不需要显式编排；QueryPlanner（LLM 拆解子查询）为可选组件，当前 Pipeline 未默认接入。

### 2.3 深度分析路径（不走Pipeline，走 Agent 链）

```
新闻解读 ──→ AgentContext(intent="news")
             → orchestrator([AnalysisAgent, ScoringAgent])
                 → AnalysisAgent._run_deep_news_chain() → call_tool("analyze_news_deep")
                     → analyze_news_text(): 抽取(并行) + KB检索 + LLM结构化分析
                     → 返回 kb_search_info (检索诊断: query/原始数/分数/阈值)
                 → ScoringAgent → evaluate + check_hallucination(mode="analysis") + report
             → 提取结果 → 返回（前端始终展示检索诊断面板）

话题调研 ──→ AgentContext(intent="deep_topic")
             → orchestrator([AnalysisAgent, ScoringAgent])
                 → AnalysisAgent._run_deep_topic_chain() → call_tool("analyze_topic_deep")
                     → analyze_topic_research(): 抓新闻 + KB检索 + LLM结构化研判
                 → ScoringAgent → 同上
             → 提取结果 → 返回
```

> ScoringAgent 是通用公共能力：任何 feature 往 metadata 塞 `scoring_source_items` / `scoring_mode` / `scoring_text` 三个字段即可接入。

---

## 3. 各模块目标定义

> 每个模块写清楚：目标是什么、边界在哪、质量基线是什么。

### 3.1 Agent 系统 (`agents/`)

**目标**：轻量级编排器，只做路由和调度，所有重活委托给 Tool。

| 模块 | 目标 | 边界 |
|------|------|------|
| `coordinator_agent.py` | 第一个 Agent，负责意图分类和链路选择 | 只做路由决策，不做数据处理 |
| `ingestion_agent.py` | 文档摄取，调用解析工具提取文本和元数据 | 只做 `call_tool()` 调用，不直接调 API |
| `analysis_agent.py` | 统一分析 Agent，根据 intent 选择工具链 | process() 不超过 80 行 |
| `scoring_agent.py` | 质量门控，评分 + 防幻觉 | 最后一个执行，评估整条链路 |

**质量基线**：Agent 的 `process()` 方法不超过 80 行（不含 `_extract_*` 路由辅助函数）。

### 3.2 Tool 系统 (`tools/`)

**目标**：所有业务逻辑、API 调用、计算、文件 IO 都在这里。Agent 通过 `self.call_tool()` 委托。

| 模块 | 目标 | 边界 |
|------|------|------|
| `core.py` | FunctionRegistry + ToolExecutor + ToolCallSession | 纯框架，不含业务逻辑 |
| `extraction_tools.py` | 指标/实体抽取（LLM-first + regex-fallback） | 返回 `_confidence` 和 `_source` |
| `document_parse_tools.py` | PDF/图片解析（闭包注入 + FunctionDef） | PyMuPDF 本地 / qwen-vl-plus 多模态 |
| `news_tools.py` | 新闻抓取（10jqka/Sina/EastMoney） | 不截断内容，保留完整文本 |
| `kline_tools.py` | K线 + 技术指标（MACD/RSI/KDJ/Bollinger） | 调 Tushare API，mock 模式返回合成数据 |
| `graph_tools.py` | LightRAG 图谱查询（Function Calling 接口） | 只查询，不写入 |
| `scoring_tools.py` | Pipeline 评分工具 | 被 ScoringAgent 调用 |
| `event_impact_tools.py` | 事件影响分析 | 关键词 fallback + LLM 增强 |
| `report_tools.py` | 报告生成 | Markdown 格式输出 |
| `coordinator_tools.py` | 协调器工具 | 意图分类、链路选择 |
| `analysis_tools.py` | 深度分析工具 | 被 services/analysis.py 调用 |

**注册流程**：tool 函数 → FunctionDef 定义 → `*_TOOLS` 列表 → `tools/core.py` 注册 → Agent 用 `call_tool()` 调用。

### 3.3 检索系统 (`retrievers/`)

**目标**：混合检索（BM25 + 向量 + RRF融合 + Rerank + 元数据过滤），支持增量索引和持久化。

| 模块 | 目标 | 边界 |
|------|------|------|
| `retriever.py` | HybridRetriever 主类 | 统一接口，内部组合 BM25+Vector |
| `bm25_engine.py` | BM25 关键词检索（jieba 分词） | 增量 `add()` 不覆盖已有索引 |
| `vector_engine.py` | ChromaDB 向量检索 | 惰性初始化，无数据时不创建 |
| `fusion.py` | RRF 融合 + Rerank | 融合在过滤之前 |
| `filters.py` | 元数据过滤 | 缩小候选集，不替代检索 |
| `chunker.py` | TextChunker 长文档切分 | 优先段落→句子→硬切，greedy 合并小段 |
| `persistence.py` | 索引持久化（JSON） | 保存 docs + embeddings + BM25 |
| `query_parser.py` | 查询扩展（同义词+概念关联） | 规则层 0ms + LLM 层可选 |
| `query_planner.py` | 查询规划（LLM 拆解复杂查询） | 简单查询跳过，LLM 失败自动降级 |
| `preprocessor.py` | 文本预处理流水线 | 清洗→相关性门控→长度过滤→分类 |
| `dictionaries.py` | 字典加载 | 外部 JSON 热扩展 |
| `dictionary_registry.py` | DictionaryRegistry 统一管理 | 10 种字典类型 |
| `lightrag_adapter.py` | LightRAG 适配器 | 同步封装，PDF/图片 → 实体关系抽取 |

**质量基线**：BM25 延迟 < 50ms，ChromaDB ANN < 200ms，总检索延迟 < 500ms。

### 3.4 防幻觉系统 (`guard/`)

**目标**：6 层透明校验，规则层 + LLM 层递进，每层分数可见。

| 模块 | 目标 | 说明 |
|------|------|------|
| `rule_layers.py` | L1-L4 规则层 | L1 来源锚定(jieba)、L2 数值一致、L3 引用完整(双模式)、L4 结构规范(双模式) |
| `llm_critique.py` | L5 LLM 质疑 | LLM 审查答案+来源，输出结构化发现 |
| `llm_assist.py` | L6 LLM 协助 | 低分时 LLM 修复问题 |
| `reflector.py` | HallucinationGuard 编排 | `check(answer, sources, mode)` → 权重归一化 |

**双模式**：`mode="rag"` 期望 `[N]` 引用 + `# Markdown`；`mode="analysis"` 期望 `【关键信号】` 等括号段落 + 文字引用。

**质量基线**：权重归一化 — 跳过的层（如 L5/L6 未触发时）不参与总分计算。

### 3.5 Pipeline (`core/`)

**目标**：5 阶段流水线调度，数据单向流动（Phase 1→2→3→4→5）。

| 模块 | 目标 | 边界 |
|------|------|------|
| `pipeline.py` | PipelineScheduler 5阶段调度 | 数据单向流动，不回溯 |
| `orchestrator.py` | Agent 编排执行 | 最大重试 1 次，延迟 0.1s |
| `agent_router.py` | 意图路由（4 种意图 + general fallback → Agent 链） | 规则匹配，不需要 LLM |
| `data_orchestrator.py` | 多池文本管理 | TextPreprocessor → DocTypeClassifier → KnowledgePool |
| `scorer.py` | PipelineScoreCard | 每阶段独立评分 |
| `factory.py` | 工厂函数 | 创建 registry + executor + agents |
| `indexer.py` | 4 阶段检索 | Clean → Extract → Retrieve → Verify |
| `base.py` | BaseAgent + AgentContext + AgentResult | 所有 Agent 的基类 |
| `protocol.py` | 协议定义 | 接口规范 |

### 3.6 LLM 层 (`llm/`)

**目标**：统一 LLM 调用入口，屏蔽重试、JSON 解析、缓存等细节。

| 模块 | 目标 |
|------|------|
| `caller.py` | LLMCaller：重试 + 平衡括号 JSON 解析 + 缓存 + 防幻觉约束 |
| `dashscope_client.py` | DashScope API 传输层 |
| `model_router.py` | ModelRouter：按任务复杂度自动选模型 |

**边界**：所有 LLM 调用必须经过 LLMCaller，不直接调 `dashscope_client`。

### 3.7 数据层

**目标**：数据源适配 + 预处理 + 持久化。

| 模块 | 目标 | 边界 |
|------|------|------|
| `rss_fetcher.py` | 新闻抓取（3 个国内 API） | 不截断内容 |
| `tushare_client.py` | Tushare K线 + 财务数据 | mock 模式返回合成 OHLCV |
| `services/persistence.py` | KB/metadata/index IO + 去重 + 备份 | 原子写入 + .bak 自动备份 |
| `services/analysis.py` | 深度分析业务逻辑（纯函数） | 不依赖 FastAPI |

### 3.8 API 层 (`api/`)

**目标**：thin HTTP shell，所有业务逻辑在 services/ 或 tools/。

| 模块 | 目标 |
|------|------|
| `ingest_router.py` | 文件预览、目录列表、文件/新闻摄取 + 预处理 + 去重 + 文件上传 |
| `analysis_router.py` | 配置(TTL缓存)、新闻/话题分析、K线端点 |
| `query_router.py` | Pipeline、Slot Fill、评分端点 |
| `kb_router.py` | KB 管理：状态、查询、构建、清理、删除 |
| `app_state.py` | 共享单例状态，惰性初始化 |

---

## 4. 前端能力映射

> 前端 4 个面板，每个面板背后接了什么后端能力，一目了然。

### 4.1 系统概览 (`panel-overview`)

| 前端区块 | 展示内容 | 背后能力 |
|----------|----------|----------|
| 统计卡片 | Agents/Tools/Tests/KB Docs/Index/Model 数量 | `GET /api/health` + `GET /api/kb/status` |
| 5-Phase Pipeline 可视化 | Fetch→Index→Process→Output→Evolve | `core/pipeline.py` 5 阶段调度 |
| Agent-Tool 协作映射 | 4 个 Agent 卡片 + 各自 Tool 列表 | `agents/` + `tools/` 注册表 |
| Hybrid Retrieval Pipeline | BM25+ChromaDB→RRF→Rerank→Filter | `retrievers/` 混合检索链 |
| 6-Layer Hallucination Guard | L1-L6 每层权重+说明 | `guard/` 规则层+LLM层 |
| KB 实时状态 | 文档数/索引状态/ChromaDB/元数据 | `GET /api/kb/status` |
| 意图路由示例 | 用户问题→自动路由 | `core/agent_router.py` 规则匹配 |

### 4.2 数据管理 (`panel-data`)

| 前端区块 | 用户操作 | 背后能力 | API |
|----------|----------|----------|-----|
| 文件导入 | 浏览目录→选择文件→深度/快速导入 | Agent链(IA→EA→Scoring) + 预处理 + 入KB | `POST /api/ingest/files` |
| 文件上传 | 拖拽 PDF/图片上传 | PDF解析(PyMuPDF) / 图片解析(qwen-vl) + 入KB | `POST /api/ingest/upload` |
| 新闻抓取 | 输入主题→抓取 N 条 | 3 源新闻(同花顺/新浪/东财) + 预处理 + 去重 | `POST /api/ingest/news` |
| KB Status | 查看文档/索引/大小/元数据 | 实时 KB 统计 | `GET /api/kb/status` |
| 知识库构建 | 一键构建索引 | BM25 + Embedding 预计算 + ChromaDB | `POST /api/build` |
| KB 管理 | 按来源分组查看/搜索/删除 | KB 文档 CRUD | `GET /api/kb/search` · `DELETE /api/kb/source/{name}` · `DELETE /api/kb/keyword/{kw}` |
| KB 内容 | 查看每篇文档详情 | 文档内容展示 | `GET /api/kb/documents/{id}` |

### 4.3 智能查询 (`panel-query`)

| 前端区块 | 用户操作 | 背后能力 | API |
|----------|----------|----------|-----|
| KB 搜索 | 输入问题→Top K 检索 | 混合检索(BM25+Vector+RRF+Rerank) | `POST /api/kb-query` |
| Pipeline 深度查询 | 输入问题→5 阶段全链路 | Fetch→Index→Process→Output→Evolve | `POST /api/pipeline` |
| K线技术分析 | 输入股票→查看技术指标 | Tushare API + MACD/RSI/KDJ/Bollinger 计算 | `POST /api/kline` |

### 4.4 深度分析 (`panel-analyze`)

| 前端区块 | 用户操作 | 背后能力 | API |
|----------|----------|----------|-----|
| 新闻解读 | 粘贴全文→结构化解析 | Agent链(Analysis→Scoring) + 指标/实体抽取 + KB检索 + LLM研判 + 防幻觉 | `POST /api/analyze/news` |
| 话题调研 | 输入话题→抓新闻→综合研判 | Agent链 + 新闻抓取 + KB检索 + LLM研判 | `POST /api/analyze/topic` |
| KB 检索诊断 | 自动展示(可折叠) | 检索词/原始数/Top5分数/阈值/诊断建议 | 含在 analyze 响应 `kb_search_info` |
| 追问对话 | 基于分析结果继续对话 | 会话管理 + 上下文累积 | `POST /api/chat/followup` |
| 学习历史 | 查看系统积累的知识 | 分析过程自动提取知识点 | `GET /api/kb/history` |

---

## 5. 扩展模式指南

> 新增功能时参考本章。每类扩展都有标准流程、代码示例和文件清单。

### 5.1 新增 Tool

```
1. 在 tools/<module>_tools.py 创建工具函数
2. 定义 FunctionDef（name, description, parameters schema）
3. 导出为 *_TOOLS 列表
4. 在 tools/core.py create_financial_registry() 注册
5. 如果工具需要 LLM，添加注入函数
6. Agent 通过 self.call_tool("tool_name", ...) 调用
```

### 5.2 新增 Agent

```
1. 在 agents/ 创建 Agent 类，继承 BaseAgent
2. 实现 can_handle(context) 和 process(context)
3. process() 只做 call_tool() 和组装 AgentResult
4. 在 core/factory.py 注册到 agent 列表
5. 在 core/agent_router.py 添加路由规则（如果需要）
```

### 5.3 新增数据源

```
1. 创建数据适配器（如 rss_fetcher.py 或 tushare_client.py）
2. 创建对应的 tool（如 news_tools.py）
3. 数据经过预处理流水线（preprocessor.py）
4. 存入 KB（retriever.add()，不覆盖已有）
```

### 5.4 新增意图

```
1. 在 core/agent_router.py 的 route() 方法添加匹配规则
2. 定义对应的 Agent 链
3. 如果需要新 Agent，按 5.2 创建
```

### 5.5 新增字典类型

```
1. 在 retrievers/dictionary_registry.py 注册新字典类型
2. 如需外部数据，在 data/dictionaries/ 添加 JSON 文件
3. retrievers/dictionaries.py 加载时自动发现
```

---

## 6. 目录树注释

```
financial_rag/
├── agents/           # 4 个轻量级 Agent（Coordinator/Ingestion/Analysis/Scoring）
├── api/              # 4 个 FastAPI Router（KB/Ingest/Analysis/Query）
├── core/             # Pipeline 调度、Agent 编排、路由、评分
├── data/             # 空目录，运行时数据在 ../../data/
├── guard/            # 6 层防幻觉（L1-L4 规则 + L5-L6 LLM）
├── llm/              # LLM 调用层（Caller + DashScope + ModelRouter）
├── retrievers/       # 混合检索（BM25 + Vector + RRF + Rerank + Filter）
├── services/         # 业务逻辑层（analysis + persistence）
├── static/           # 前端（HTML + 9 JS模块 + 6 CSS）
├── tools/            # 11 个工具模块（32 个注册工具）
├── config.py         # 所有配置
├── main.py           # CLI 入口
├── mock_data.py      # Mock 模式数据
├── prompts.py        # 所有 LLM prompt
├── rss_fetcher.py    # 新闻抓取
├── slot_filler.py    # 槽位填充
├── templates.py      # Markdown 模板
├── tushare_client.py # Tushare API 客户端
└── web.py            # FastAPI 应用入口

data/
├── dictionaries/     # 外部字典 JSON（股票映射、同义词等）
├── financial/        # 财务数据缓存
└── knowledge_base/   # KB 文档 + 索引 + 新闻 + 图谱

tests/                # 测试（smoke + 单元 + 集成）
docs/                 # 文档（ARCHITECTURE + PROJECT_QA + 截图）
experiments/          # 实验代码（LightRAG PoC 等）
```

---

## 7. 关键概念速查表

| 概念 | 说明 |
|------|------|
| Agent 链 | Coordinator → Ingestion → Analysis → Scoring，意图路由决定走哪几个 |
| 5 阶段 Pipeline | Fetch → Index → Process → Output → Evolve |
| Function Calling | Agent 通过 `self.call_tool()` 委托工具执行，不直接调 API |
| 混合检索 | BM25(关键词) + Vector(语义) → RRF 融合 → Rerank → 元数据过滤 |
| 增量索引 | `retriever.add()` 不覆盖已有 KB；`retriever.index()` 全量重建 |
| 防幻觉双模式 | RAG 模式 `[N]` 引用 / 分析模式 `【】` 段落，权重归一化 |
| 预处理流水线 | 清洗 → 相关性门控 → 最小长度 → 文档分类 → 去重 |
| LightRAG | 知识图谱，只接受 PDF/图片解析结果，JSON + GraphML 文件存储 |
| QueryPlanner | LLM 拆解复杂查询为子查询，独立组件，当前 Pipeline 未默认接入 |
| 数据角色 | 文件=知识(入KB)，新闻=元数据+入KB(预处理后)，K线=实时(不入KB) |
