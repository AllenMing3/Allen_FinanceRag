# Agent & Tools 速查手册

> 自己看的版本。面试时能快速说出"哪个 Agent 调哪些 Tool、为什么这么设计"。

---

## 一、Agent 全景

```
用户查询
  │
  ▼
AgentRouter (core/agent_router.py)
  │  意图分类: kline / event_impact / report / news / general
  │  选择 Agent 链
  │
  ▼
PipelineScheduler 编排执行
  │
  ├── IngestionAgent (可选，report/news/general 链前置)
  ├── AnalysisAgent (核心分析引擎)
  └── ScoringAgent (所有链末端)
```

---

## 二、4 个 Agent 详解

### 1. IngestionAgent（数据摄取）

**职责**：把原始文档变成结构化数据，存入 `context.parsed_data`

**调用链**：
```
对每个文档:
  → call_tool("extract_document_metadata")   # 抽取: 公司名、日期、文档类型
  → call_tool("detect_document_type")        # LLM 判断: 年报/季报/公告/新闻/研报
```

**输出**：
- `context.parsed_data = [{"text": "...", "meta": {"source": "xxx", "doc_type": "annual_report", ...}}]`
- `intermediate_findings` 中记录每个文件的处理状态

**特点**：
- 文件级粒度，逐文件处理（不是批量）
- 后台线程执行，前端轮询 `/api/ingest/progress` 看进度
- 如果 LLM 不可用，doc_type 回退到启发式判断

---

### 2. AnalysisAgent（统一分析）

**职责**：根据 `context.metadata["intent"]` 选择不同工具链

**5 条工具链**：

| intent | 方法 | 调用的 Tools | 输出 |
|--------|------|-------------|------|
| `kline` | `_run_kline_chain()` | `analyze_kline` → `generate_kline_analysis` | 技术指标 + LLM 研判 |
| `event_impact` | `_run_event_chain()` | `fetch_date_events` → `fetch_kline_context`(可选) → `assess_event_impact` | 事件列表 + 利好/利空 |
| `news` (有parsed_data) | `_run_deep_news_chain()` | `analyze_news_deep` | 多维影响 + 关键信号 + 风险 |
| `report`/`general` | `_run_extraction_chain()` + `_generate_report()` | `extract_financial_metrics` → `extract_entities` → `generate_search_queries` → `synthesize_report` | 抽取报告 |
| `report` (话题调研) | `_run_deep_topic_chain()` | `analyze_topic_deep` | 子话题 + 参与者 + 情绪趋势 |

**路由逻辑**（`process()` 方法）：
```python
if intent == "kline":
    findings = self._run_kline_chain(context)
elif intent == "event_impact":
    findings = self._run_event_chain(context)
elif intent == "news" and context.parsed_data:
    return self._run_deep_news_chain(context)  # 深度分析
else:
    findings = self._run_extraction_chain(context)  # 通用抽取
```

**核心设计**：
- Agent 不含业务逻辑，所有计算/获取/LLM 调用都通过 `call_tool()`
- `_render_markdown()` 统一把工具结果渲染成可读 Markdown
- `_render_structured_news/topic()` 处理深度分析的结构化输出

---

### 3. ScoringAgent（全链路评分）

**职责**：对 Pipeline 各阶段结果打分 + 防幻觉校验

**调用链**：
```
→ call_tool("evaluate_pipeline_quality")    # 各阶段打分 (fetch/index/process/output)
→ call_tool("check_hallucination")          # 六层防幻觉校验
→ call_tool("generate_score_report")        # 生成可读评分报告
```

**输入来源**：
- `context.metadata["fetched_data"]` — 阶段1数据
- `context.metadata["retrieved_items"]` — 阶段2检索结果
- `context.metadata["fetch_elapsed_ms"]` / `index_elapsed_ms` — 时间指标
- `context.intermediate_findings` — 各 Agent 的 `{"stage": "xxx", "success": True/False}`

**输出**：
- `data["pipeline_scores"]` — 各阶段分数 + 等级(A/B/C/D/F)
- `data["hallucination_check"]` — 风险等级(low/medium/high)
- `data["any_agent_failed"]` — 是否有上游 Agent 失败
- `success` = 评分过程是否完成（不是分数好不好看）

---

### 4. CoordinatorAgent（路由协调）

**职责**：意图分类 + Agent 链选择

**调用链**：
```
→ call_tool("classify_query_intent")   # 返回 intent + confidence
→ call_tool("select_agent_chain")      # 根据 intent 选择 agent 链
```

**当前状态**：**死代码**。已注册在 factory 但没有任何 chain 引用它。路由逻辑已由 `PipelineScheduler._route_query()` + `AgentRouter` 完成。

---

## 三、28 个 Tools 完整清单

### 检索类（1 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `search_financial_data` | core.py | 知识库语义搜索 | Function Calling 会话 |

### 分析计算类（4 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `calculate_growth_rate` | core.py | 同比增长率 | Function Calling |
| `calculate_financial_ratio` | core.py | 财务比率(毛利率/ROE等) | Function Calling |
| `compare_metrics` | core.py | 两家公司指标对比 | Function Calling |
| `summarize_financials` | core.py | 指标汇总成文本 | Function Calling |

### 数据获取类（3 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `fetch_stock_news` | core.py | 拉取股票新闻 | Function Calling |
| `fetch_financial_news` | core.py | 拉取财经新闻 | Function Calling |
| `fetch_announcements` | core.py | 拉取上市公司公告 | Function Calling |

### 抽取类（5 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `extract_document_metadata` | extraction_tools.py | 抽取公司名/日期/金额 | IngestionAgent |
| `extract_entities` | extraction_tools.py | 抽取实体(公司/人物/模型/芯片...) | AnalysisAgent (general) |
| `extract_financial_metrics` | extraction_tools.py | 抽取财务指标(营收/利润/毛利率...) | AnalysisAgent (general) |
| `extract_document_metadata` | extraction_tools.py | 文档元数据 | IngestionAgent |
| `generate_search_queries` | extraction_tools.py | 生成检索查询(辅助检索) | AnalysisAgent (general) |
| `detect_document_type` | extraction_tools.py | LLM判断文档类型 | IngestionAgent |

### 新闻类（1 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `fetch_news_report` | news_tools.py | 搜索新闻 + 保存为 Markdown 报告 | Function Calling |

### K线类（4 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `fetch_etf_kline_report` | kline_tools.py | ETF K线 + 统计分析 + Markdown | Function Calling |
| `fetch_kline_context` | kline_tools.py | 获取日期附近的K线上下文 | AnalysisAgent (event_impact) |
| `analyze_kline` | kline_tools.py | 技术指标计算(MACD/RSI/KDJ/Boll) | AnalysisAgent (kline) |
| `generate_kline_analysis` | kline_tools.py | LLM K线趋势研判 | AnalysisAgent (kline) |

### 事件影响类（2 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `fetch_date_events` | event_impact_tools.py | 按日期获取事件列表 | AnalysisAgent (event_impact) |
| `assess_event_impact` | event_impact_tools.py | 事件利好/利空评估 + 影响因子 | AnalysisAgent (event_impact) |

### 评分类（3 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `evaluate_pipeline_quality` | scoring_tools.py | Pipeline 各阶段打分 | ScoringAgent |
| `check_hallucination` | scoring_tools.py | 六层防幻觉校验 | ScoringAgent |
| `generate_score_report` | scoring_tools.py | 生成可读评分报告 | ScoringAgent |

### 调度类（2 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `classify_query_intent` | coordinator_tools.py | 意图分类 | CoordinatorAgent (死代码) |
| `select_agent_chain` | coordinator_tools.py | Agent 链选择 | CoordinatorAgent (死代码) |

### 报告类（1 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `synthesize_report` | report_tools.py | LLM 综合报告生成 | AnalysisAgent (general) |

### 深度分析类（2 个）

| Tool | 文件 | 功能 | 被谁调 |
|------|------|------|--------|
| `analyze_news_deep` | analysis_tools.py | 新闻深度分析(多维影响+信号+风险) | AnalysisAgent (news) |
| `analyze_topic_deep` | analysis_tools.py | 话题调研(子话题+参与者+情绪趋势) | AnalysisAgent (report/话题) |

---

## 四、Tool 注入机制

Tool 层不直接 import LLM / Retriever，而是通过**模块级引用 + 注入函数**：

```
create_financial_registry(retriever, llm)
  │
  ├── inject_extraction_llm(llm)          # extraction_tools 的 LLM
  ├── inject_event_llm(llm)               # event_impact_tools 的 LLM
  ├── inject_kline_llm(llm)               # kline_tools 的 LLM
  ├── inject_report_llm(llm)              # report_tools 的 LLM
  ├── inject_analysis_deps(llm, retriever) # analysis_tools 的 LLM + Retriever
  └── set_retriever(retriever)            # search_financial_data 的检索器
```

**设计原则**：Agent 只调 `self.call_tool("xxx", ...)`，不关心 Tool 内部用什么 LLM 或数据源。新增能力只需：
1. 写工具函数 + FunctionDef
2. 在 `create_financial_registry()` 注册 + 注入依赖
3. Agent 通过 `self.call_tool("new_tool", ...)` 调用

---

## 五、数据流：从查询到回答

```
用户输入 "商汤科技2024年营收增长了多少？"
  │
  ▼
[Phase 1: Fetch] Function Calling 选择数据源
  → fetch_financial_news("商汤科技") 或 search_financial_data(...)
  │
  ▼
[Phase 2: Index] 入 RAG 索引 + 混合检索
  → TextChunker 切分 → BM25 + Vector → RRF 融合 → qwen3-rerank 重排
  │
  ▼
[Phase 3: Process] AgentRouter 路由 + Agent 链执行
  → intent="report", chain=[IngestionAgent, AnalysisAgent, ScoringAgent]
  → IngestionAgent: parsed_data = [{text, meta}]
  → AnalysisAgent(general): extract_metrics → extract_entities → synthesize_report
  → ScoringAgent: evaluate_pipeline_quality → check_hallucination
  │
  ▼
[Phase 4: Output] SlotFiller 填充模板
  → 用检索到的指标填充 fin 模板槽位
  │
  ▼
[Phase 5: Evolve] PipelineScoreCard 全链路评分
  → 各阶段分数 + 诊断 + 改进建议
```

---

## 六、面试时可说的设计亮点

1. **Agent 只是编排者**：所有业务逻辑在 Tool 层，Agent 只做路由决策 + 工具调用
2. **深度分析桥接**：Service 层的结构化分析能力通过 `analysis_tools.py` 暴露给 Agent 链，保证两条路径输出一致
3. **失败传播可见**：`intermediate_findings` 带 `success` 字段，ScoringAgent 能感知上游 Agent 失败
4. **LLM 注入解耦**：Tool 通过 `_llm_ref` 字典持有 LLM 引用，由 `create_financial_registry` 统一注入，Tool 本身无状态
5. **新增能力零侵入**：加新 Tool 不碰 Agent 代码，只改注册 + 注入

---

## 七、已知问题 / 可改进点

| 问题 | 状态 | 说明 |
|------|------|------|
| CoordinatorAgent 是死代码 | 已知 | 注册了但没有 chain 用它，路由已由 PipelineScheduler 完成 |
| 双重评分 | 已知 | `_phase_evolve()` 和 ScoringAgent 各自独立打分 |
| `report` intent 未走深度分析 | 设计决策 | 财报分析场景保留 extraction 路径 |
| `_run_deep_topic_chain` 是备用路径 | 设计决策 | 目前只从 web 端直接调 service，agent 链暂未使用 |
