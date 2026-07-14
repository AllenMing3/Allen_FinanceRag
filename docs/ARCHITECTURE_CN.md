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

### 2.2 查询路径

```
用户问题
  → QueryParser（查询扩展：同义词+概念关联）
  → QueryPlanner（LLM拆解复杂查询为子查询，简单查询跳过）
  → Phase 1: Fetch（新闻/K线实时数据）
  → Phase 2: Index（增量索引新数据，不覆盖已有KB）
  → Phase 3: Process（AgentRouter选链路 → Agent链执行）
  → Phase 4: Output（SlotFiller + 组装最终回答）
  → Phase 5: Evolve（ScoreCard评分 + HallucinationGuard防幻觉）
  → 返回给用户
```

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
| `agent_router.py` | 意图路由（5 种意图 → Agent 链） | 规则匹配，不需要 LLM |
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

## 4. 扩展模式指南

> 新增功能时参考本章。每类扩展都有标准流程、代码示例和文件清单。

### 4.1 新增 Tool

```
1. 在 tools/<module>_tools.py 创建工具函数
2. 定义 FunctionDef（name, description, parameters schema）
3. 导出为 *_TOOLS 列表
4. 在 tools/core.py create_financial_registry() 注册
5. 如果工具需要 LLM，添加注入函数
6. Agent 通过 self.call_tool("tool_name", ...) 调用
```

### 4.2 新增 Agent

```
1. 在 agents/ 创建 Agent 类，继承 BaseAgent
2. 实现 can_handle(context) 和 process(context)
3. process() 只做 call_tool() 和组装 AgentResult
4. 在 core/factory.py 注册到 agent 列表
5. 在 core/agent_router.py 添加路由规则（如果需要）
```

### 4.3 新增数据源

```
1. 创建数据适配器（如 rss_fetcher.py 或 tushare_client.py）
2. 创建对应的 tool（如 news_tools.py）
3. 数据经过预处理流水线（preprocessor.py）
4. 存入 KB（retriever.add()，不覆盖已有）
```

### 4.4 新增意图

```
1. 在 core/agent_router.py 的 route() 方法添加匹配规则
2. 定义对应的 Agent 链
3. 如果需要新 Agent，按 4.2 创建
```

### 4.5 新增字典类型

```
1. 在 retrievers/dictionary_registry.py 注册新字典类型
2. 如需外部数据，在 data/dictionaries/ 添加 JSON 文件
3. retrievers/dictionaries.py 加载时自动发现
```

---

## 5. 目录树注释

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

## 6. 关键概念速查表

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
| QueryPlanner | LLM 拆解复杂查询为子查询，简单查询跳过，LLM 失败自动降级 |
| 数据角色 | 文件=知识(入KB)，新闻=元数据+入KB(预处理后)，K线=实时(不入KB) |
