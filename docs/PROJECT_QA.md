# 项目面试 Q&A — FinRAG Agent 系统

> 基于 commit 历史整理，涵盖架构决策、踩坑经历、工程演进。  
> 目的：熟悉自己的项目，能在面试中讲清楚"做了什么、为什么这么做、遇到了什么问题、怎么解决的"。

---

## Q1: 简单介绍一下这个项目？

这是一个**金融领域的 RAG（Retrieval-Augmented Generation）智能体系统**，核心功能：
- **新闻解读**：粘贴新闻文本 → 多维影响分析（行业/公司/技术/市场）+ 关键信号 + 风险提示
- **话题调研**：输入话题关键词 → 子话题聚类 + 关键参与者 + 情绪趋势 + 反向信号 + 投资启示
- **K线分析**：输入股票代码 → 技术指标计算 + LLM 趋势研判
- **知识库管理**：文件导入/索引/去重/持久化，支持增量学习

技术栈：**FastAPI + DashScope (Qwen) + ChromaDB (ANN) + 自研 HybridRetriever (BM25 + ChromaDB + RRF) + 4 Agent 协作**。

---

## Q2: 为什么要做 Agent 整合？7个 Agent 减到4个是怎么想的？

**问题**：最初有 7 个 Agent（KLineAgent、ExtractionAgent、EventImpactAgent、ReportAgent 等），每个只做一件固定的事。

**发现**：
- KLineAgent 本质就是 `fetch_kline → analyze_kline`，是**固定的工具链调用**，不需要决策
- ExtractionAgent 和 ReportAgent 职责高度重叠——都是"提取指标+生成报告"
- Agent 数量多了，chain 组合爆炸，测试覆盖也跟不上

**决策**：
- 合并为 **4 个 Agent**：IngestionAgent（数据摄入）、AnalysisAgent（统一分析）、ScoringAgent（质量评分）、CoordinatorAgent（路由协调）
- 所有"固定工具链"下沉为 AnalysisAgent 内部的方法（`_run_kline_chain`、`_run_event_chain`、`_run_extraction_chain`）
- Agent 只做**意图路由 + 工具编排**，不做重计算

**收益**：代码量减少 ~30%，测试更集中，新人上手更快。

---

## Q3: MCP 数据源为什么要移除？

**问题**：最初用 MCP（Model Context Protocol）客户端接入外部数据源（RSS 新闻、股票数据），但 MCP 是第三方协议，调试困难、错误信息不透明。

**踩坑**：
- MCP 客户端连接不稳定，超时后没有清晰的错误
- 国内新闻 RSS 经常被封，获取率很低
- 调试时需要同时看 MCP 协议层和业务层，排查成本翻倍

**决策**：
- 移除 MCP，改为**直连国内新闻 API**（如天行数据）+ Tushare 获取 K线
- 新闻用 httpx 直接调，错误处理清晰，限流逻辑自己控

---

## Q4: BM25 检索踩过什么坑？

**问题**：最初用 LlamaIndex 内置的 BM25，结果中文分词效果极差。

**根因**：
- LlamaIndex 的 BM25 按空格分词，中文没有空格 → 整段文本被当成一个 token
- 导致 BM25 检索几乎无召回，全靠向量检索兜底

**解决**：
- 迁移到 `rank_bm25` 库，配合 `jieba` 分词
- 加入**文档长度归一化**：BM25 默认偏向长文档，加了 `b` 参数调节
- 加了**索引去重**：同一篇文档被多次索引后，BM25 会返回重复结果，用 doc_id 去重

**收益**：BM25 召回率从 ~5% 提升到 ~60%，RRF 融合后整体相关性显著提升。

---

## Q5: RRF（Reciprocal Rank Fusion）融合是怎么做的？

**背景**：混合检索 = BM25（关键词匹配）+ ChromaDB ANN（语义相似）。两个检索器返回的文档排序不同，需要融合。

**实现**：
```
RRF_score(doc) = Σ 1 / (k + rank_i)   # k=60，rank_i 是该 doc 在第 i 个检索器中的排名
```
- BM25 和 Vector 各贡献一个排名分
- 两个检索器都返回的文档得分更高（共识奖励）
- 最后加 qwen3-rerank 重排

**关键细节**：融合时**必须用新的 RRF score 替换原始 score 字段**，否则 UI 显示的还是单个检索器的分数，排序会乱。

---

## Q6: LLM 调用不稳定怎么办？

**问题**：DashScope API 偶尔超时、返回格式不一致、JSON 解析失败。

**方案：LLMCaller 统一调用层**（commit `ac9c94e`）：
- **自动重试**：超时/5xx 错误自动重试 2 次，指数退避
- **响应缓存**：相同输入缓存结果（TTL 5分钟），减少 API 消耗
- **JSON 解析**：`call_json()` 方法自动尝试 JSON 解析，用平衡括号匹配算法处理嵌套对象，失败后 fallback 到正则提取
- **统一接口**：所有 LLM 调用都经过 LLMCaller，不直接调 DashScope SDK

**设计原则**：工具层（`tools/`）不关心 LLM 怎么调，只通过注入的 `_llm_ref` 获取结果。

---

## Q7: 防幻觉（Hallucination Guard）怎么做的？

**四层透明校验**（`guard/reflector.py`），每层做实、分数可见：

1. **L1 来源锚定**（weight 0.35）：jieba 分词后逐句检查 token 重叠率，每句能否追溯到某篇检索源。阈值 0.15。
2. **L2 数值一致**（weight 0.25）：提取 answer 中的「数字+单位」对，交叉比对 source 中是否有相同数字。防止编造营收/增长等关键数据。
3. **L3 引用完整**（weight 0.20）：`[N]` 标记存在且 N 对应有效来源编号。防止“伪引用”。
4. **L4 结构规范**（weight 0.20）：输出是否包含预期段落（摘要/要点/风险等）。

**用户可见**：每层的得分、通过/未通过、未锚定的句子都会出现在最终输出的防幻觉报告中，不是藏在 metadata 里。

**集成点**：ScoringAgent 在 Phase 5 (Evolve) 调用 `check_hallucination` 工具，评分卡 + 防幻觉报告一起写入最终输出。

**为什么从 6 层改成 4 层**：原来的 6 层有几层是“凑数”的（完整性、事实核查）——它们和 L1 来源锚定高度重叠，且检查逻辑不够具体。重写后每层有明确的检查目标和独立的分数，而不是“看起来层数多但实际效果含糊”。

---

## Q8: Agent 链的深度断裂是怎么发现的？怎么修的？

**问题**：新闻解读和话题调研的"深度分析"只在 Service 层（`services/analysis.py`）实现了结构化输出（多维影响、关键信号、子话题聚类），但 Agent 链走的是旧路径（extract_metrics + synthesize_report），输出是浅层文本。

**发现方式**：做 Code Review 时发现 `/api/analyze/news` 返回结构化 JSON，但 `/api/pipeline` 走 Agent 链时返回的是旧格式——**同一个功能，两条路径输出质量不一致**。

**修复（Plan A）**：
- 创建 `tools/analysis_tools.py`，封装 `analyze_news_deep` 和 `analyze_topic_deep` 工具
- 在 `create_financial_registry()` 注册并注入 LLM + Retriever
- `AnalysisAgent.process()` 检测 `news` intent + 有 `parsed_data` 时，调用深度分析工具
- 添加 `_render_structured_news()` 和 `_render_structured_topic()` Markdown 渲染器

---

## Q9: ScoringAgent 之前有什么 bug？

**三个连锁问题**：
1. **无法感知上游失败**：`intermediate_findings` 没有 `success` 字段，ScoringAgent 硬编码 `success=True`，永远报 100% agent 成功率
2. **chain 失败不中断**：Orchestrator 在 `enable_retry=True`（默认配置）时，agent 失败后不 break，下游 agent 用脏数据继续跑
3. **自身永远 success=True**：即使评分过程出错，ScoringAgent 仍返回 `success=True`

**修复**：
- AnalysisAgent 在 `intermediate_findings` 中携带 `{"success": True/False}`
- ScoringAgent 读取 `finding.get("success", True)` 而非硬编码
- Orchestrator 失败时始终 log warning（而非静默继续）
- ScoringAgent 的 `success` 由"评分是否完成"决定（而非"分数是否好看"）

---

## Q10: 知识库持久化踩过什么坑？

**问题**：最初每次启动都重建索引，几百篇文档重建耗时 30+ 秒。

**解决**（commit `0120f0e`）：
- **索引持久化**：首次构建后保存到 `output/index_cache.pkl`，下次启动直接加载
- **Doc ID 去重**：用 `hash(doc_text[:200])` 生成唯一 ID，防止同一文档重复入库
- **备份轮转**：索引更新前自动备份旧版本，最多保留 3 个备份
- **学习历史**：记录每次增量学习的时间、新增文档数、触发来源

**额外收益**：去重后知识库体积减少了 ~40%（之前每次导入都重复添加）。

---

## Q11: 全链路打分（Pipeline ScoreCard）是什么？

每个 Pipeline 阶段独立评分，精确诊断薄弱环节：

| 阶段 | 评分维度 |
|---|---|
| 数据获取 | 是否获取到数据、数据条数 |
| RAG 检索 | BM25/ChromaDB 命中率、RRF 共识度、Rerank 高相关比例 |
| Multi-Agent | 各 Agent 成功率、工具调用成功率 |
| 槽位输出 | 模板填充率 |
| 防幻觉 | 四层透明校验（来源锚定 + 数值一致 + 引用完整 + 结构规范） |

最终输出加权总分 + 等级（A/B/C/D/F）+ 最薄弱 3 个环节的诊断建议。

**面试时可说**：这个设计的灵感来自 Datadog 的 SLI/SLO 理念——不是给一个笼统的"好不好"，而是精确告诉你**哪个环节不好、为什么不好、怎么改**。

---

## Q12: 数据导入的用户体验做了哪些改进？

**之前**：用户只能"导入整个目录"，没有选择，没有预览，不知道导入了什么。

**之后**（commit `f6d6a1e`）：
- **文件勾选**：每个文件带 checkbox + "全选"
- **内容预览**：点击文件名可预览前 20 行
- **分析模式切换**：深度分析（LLM 抽取指标+实体）vs 快速导入（跳过 LLM）
- **新闻条数选择器**：10/20/30/50 条
- **关键词过滤**：只导入匹配关键词的新闻
- **自定义目录**：直接输入路径，不再藏在折叠面板里

---

## Q13: 项目的测试策略是什么？

**四层测试**（507 tests passing）：
1. **单元测试**：每个模块独立测试（test_agents.py, test_analysis.py, test_query_parser.py）
2. **集成测试**：Agent 链端到端（test_new_agents.py, test_orchestrator_merge.py）
3. **Smoke 测试**：Web API 全链路（test_smoke.py），验证每个 endpoint 不 crash
4. **计算测试**：技术指标计算正确性（test_tushare_compute.py — MACD/RSI/KDJ/Bollinger 结构、值域、公式验证）

**原则**：
- Mock 只 mock **外部数据源**（Tushare API、新闻 API），不 mock LLM/Rerank/Embedding
- 每次加新功能必须加对应测试
- Smoke test 覆盖所有 `/api/*` 端点，确保前端不会因为后端 crash 而白屏

---

## Q14: 如果面试官问"这个系统还有什么可以改进的"？

诚实回答（展示工程思维）：
1. **双重评分问题**：`_phase_evolve()` 和 ScoringAgent 各自独立评分，可能不一致。应统一为一条路径
2. **流式输出**：目前 LLM 分析是同步等待完整结果，可以改为 SSE 流式返回提升用户体验
3. **生产级部署**：目前是单进程 + 内存状态，生产环境需要 Redis 状态管理 + 水平扩展
4. **真实数据覆盖**：K线和技术指标目前主要通过 mock 测试覆盖，真实 Tushare API 的集成测试需要有效的 Token（120+ 积分）

---

## Q15: 项目里最有技术深度的一个点是什么？

**Agent-Tool 委托架构 + Function Calling 闭环**：

```
用户查询 → AgentRouter 识别意图 → PipelineScheduler 选择 Agent 链
  → AnalysisAgent 调用 call_tool("analyze_news_deep")
    → analysis_tools.py 封装 service 层调用
      → LLMCaller.call_json() 结构化输出
        → HallucinationGuard 防幻觉校验
          → ScoringAgent 全链路评分
            → Markdown 渲染返回前端
```

每个环节职责单一、可独立测试、可替换。新增能力只需：
1. 写工具函数 → 2. 注册 FunctionDef → 3. Agent 调用 `self.call_tool("xxx")`

**不需要修改 Agent 本身**——这就是"Agent 做编排，Tool 做计算"的核心价值。

---

## Q16: HallucinationGuard 静默失败是怎么发现的？

**问题**：终端日志里偶尔出现 `[WARNING] LLM 报告生成失败: 'str' object has no attribute 'get'`，但报告仍然生成了——用户看不到任何异常。

**根因**：`synthesize_report` 在调用 `HallucinationGuard.precheck()` 前，先把 source dict 提取成了 `List[str]`（只取前 200 字符），但 Guard 内部 `_l1_source_verification()` 对每个 source 调 `.get("text", "")`——字符串没有 `.get()` 方法。

**发现方式**：不是用户报错，是看终端日志时发现的 WARNING。因为外层有 `except Exception` 捕获后走 fallback，所以报告照常生成，但**防幻觉校验完全跳过了**。

**修复**：不再预提取文本，直接把完整的 source dict 列表传给 `precheck()`。

**教训**：broad `except Exception` 是静默 bug 的温床——错误被吞掉，系统继续运行，看起来一切正常但关键防护失效。应该在 catch 时至少 log warning + 在返回结果中标记 `guard_skipped=True`。

---

## Q17: 前端前后端字段不一致怎么发现的？

**问题**："学习历史"面板报错 `Cannot read properties of undefined (reading 'startsWith')`。

**根因**：前端 JS 在读 `item.source`（KB 文档旧格式：`"analysis:news:xxx"`），但后端 `/api/kb/history` 返回的是学习记录 JSONL 格式，字段是 `item.analysis_type` + `item.topic`。还尝试读 `item.preview`——这个字段根本不存在。

**发现方式**：用户直接看到前端报错。

**修复**：前端改为读 `item.analysis_type` 判断类型、`item.topic` 取话题名，删除不存在的 `preview` 渲染行，所有字段加 `|| ''` 防御。

**教训**：后端数据格式演进时没有同步更新前端。应该在改 `append_learning_record()` 返回结构时，同步检查所有消费方（前端 JS + API 测试）。

---

## Q18: 为什么把一些功能从前端移除？

**决策背景**：前端曾经有 6 个标签页（系统概览、导入数据、构建知识库、RAG 查询、智能分析、分析工具），还暴露了“槽位填充”和“检索链路诊断”等内部机制。问题：
- **槽位填充**（SlotFiller）：是 Pipeline Phase 4 (Output) 的内部机制，用户不应该关心
- **检索打分**（ScoreCard）：是 Pipeline Phase 5 (Evolve) 的内部机制，已经在每条链末端自动运行
- **新闻搜索**（分析工具面板）：和“导入数据”的新闻抓取调用同一个 `run_news_pipeline()`，纯重复
- **6 个 tab 太乱**：内部架构泄漏到用户界面，用户看到一堆技术细节而不知道怎么用

**修复**：
- 前端精简为 **4 个标签页**：系统概览、数据管理（合并导入+构建+管理）、智能查询（合并 RAG+K线）、深度分析
- Agent 架构页面只展示 4 个 Agent 角色卡片 + 意图路由示例，不展示工具名、I/O 规格、Function Calling、LLMCaller 等内部细节
- 所有内部架构信息移到 `docs/ARCHITECTURE.md`

**原则**：基础设施先确保在后端充分集成，再决定是否暴露 UI。UI 暴露内部机制 = 架构泄漏。

---

## Q19: 向量检索为什么用 ChromaDB？

**问题**：原来的向量检索是内存全量 cosine 相似度——文档少时还行，文档多了线性扫描越来越慢。

**决策**：引入 ChromaDB 作为向量数据库：
- **HNSW ANN 索引**：近似最近邻搜索，不用全量算 cosine
- **PersistentClient**：向量持久化到磁盘（`data/knowledge_base/chroma/`），服务重启不用重新 embed
- **内容哈希 MD5 ID**：每个文档用 `md5(content[:200])` 作为稳定 ID，跨 add/remove 操作不变
- **删除同步**：`HybridRetriever.remove()` 同时删除内存 docs + ChromaDB 中对应 ID 的向量，不用全量重建

**回退机制**：
- 无 API Key → VectorEngine 退化为 Jaccard 相似度（不需要 embedding）
- 有 API Key 但 ChromaDB 初始化失败 → brute-force cosine 回退

---

## Q20: 中文检索效果不好怎么办？

**问题**：BM25 用 jieba bigram 分词，粒度太粗，中文检索精度不够。

**解决**：
- **Trigram 分词**：jieba 切分后，≥4 字的中文片段进一步拆成 trigram，保留 2-8 字的完整 segment。更细的 token 粒度让 BM25 召回更准
- **索引前清洗**：`TextPreprocessor` 去模板化文本 + 段落去重，默认开启
- **MD5 稳定 ID**：融合时用 `md5(content)` 替代 Python 内置 `hash()`（重启后值变），保证 RRF 去重结果稳定

**效果**：中文检索相关性明显提升，重复文档问题彻底解决。

---

## Q21: 性能优化做了哪些？

**审计驱动**：不是凭感觉优化，而是先审计全代码库，找出“暴力”模式再定点修复。

| 优化项 | 原来 | 现在 | 文件 |
|---|---|---|---|
| 关键词扫描 | `for kw in list: if kw in text` O(n×m) | 预编译 `re.compile()` + `findall()` O(n) | `analysis.py`, `extraction_tools.py` |
| 文档类型检测 | ~100 次 `kw in text` 循环 | 单次 regex 扫描 + frozenset 交集 | `extraction_tools.py` |
| 指标别名排除 | O(n×m×k) 三层嵌套循环 | O(1) set 查找 | `extraction_tools.py` |
| 抽取并行化 | 指标抽取 + 实体抽取串行 | `ThreadPoolExecutor(2)` 并行 | `analysis_agent.py` |
| Pipeline 跳过 | Phase 4 槽位填充始终执行 | Agent 已生成 >50 字时跳过 SlotFiller | `pipeline.py` |

**原则**：所有预编译/预计算放在模块级别（import 时执行一次），函数调用时零开销。

---

## Q22: LLM 调用层还做了什么优化？

**平衡括号 JSON 解析**：原来 `call_json()` 用贪婪正则提取 JSON，遇到嵌套对象会截断。改用平衡括号匹配算法，正确处理字符串内的转义字符和嵌套 `{}`，解析成功率明显提升。

**SlotFiller 也走 LLMCaller**：槽位填充的 LLM 调用之前是裸调 `llm.chat()`，没有重试/缓存/输入校验。统一后所有 LLM 调用都经过 LLMCaller，TTFT 也从固定公式改为实测延迟。

---

## Q23: 查询扩展 (Query Expansion) 是怎么做的？

**问题**：用户输入 "英伟达" 时，BM25 只搜 "英伟达" 三个字，但知识库文档可能写的是 "NVIDIA"、"NVDA"、"黄仁勋"。短查询召回率很低。

**解决**：规则优先 + LLM 增强的两层查询扩展：

| 层 | 策略 | 权重 | 延迟 |
|---|---|---|---|
| 同义词扩展 | 35 组双向同义词（"英伟达" ↔ "NVIDIA" ↔ "NVDA"），O(1) 查找表 | 1.5 | 0ms |
| 概念关联 | 18 个行业概念单向关联（"芯片" → [半导体, 光刻, 晶圆]） | 0.6 | 0ms |
| LLM 增强 | 短查询 <15 字且规则扩展 <2 词时，LLM 补充 2-3 个搜索词 | - | ~500ms |

**数据流**：
- `QueryParser._expand_query()` 扫描查询中的词，查 `SYNONYM_LOOKUP` 和 `CONCEPT_MAP`
- 扩展词添加到 `result.keywords` 并生成 `result.expanded_query`（原 query + 扩展词拼接）
- `HybridRetriever.search()` 中 BM25 用扩展后加权关键词，ChromaDB 用 `expanded_query` 做语义检索
- 可选 `llm_rewrite_query()` 对短查询做 LLM 语义补充

**示例**：
```
输入: "英伟达芯片"
  → 同义词: NVIDIA, NVDA (weight=1.5)
  → 概念: 半导体, 光刻, 晶圆, 封装, 制程 (weight=0.6)
  → expanded_query: "英伟达芯片 NVIDIA NVDA 半导体 光刻 晶圆 封装 制程"
  → BM25 用以上加权词搜索，ChromaDB 用拼接后的语义查询
```

---

## 关键数字（面试时可引用）

| 指标 | 数据 |
|---|---|
| 总 commit 数 | ~60 |
| Agent 数量 | 4（从 7 精简） |
| 注册工具 | 27 个，跨 9 个模块 |
| 测试覆盖 | 521 tests（21 个测试文件） |
| 知识库文档 | ~500+ 篇（去重后） |
| 检索延迟 | BM25 < 50ms，ChromaDB ANN < 200ms |
| 查询扩展 | 35 组同义词 + 18 个概念关联，规则层 0ms + LLM 层可选 |
| 全链路评分 | 4 层防幻觉 + 5 阶段打分 |
| API 架构 | 4 个 FastAPI Router（KB / Ingest / Analysis / Query） |
| 意图路由 | 5 种意图（kline / event_impact / report / news / general） |
