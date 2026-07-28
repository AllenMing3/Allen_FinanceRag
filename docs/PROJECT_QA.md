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

**六层透明校验**（`guard/` 模块），规则层 + LLM 层递进，每层分数可见：

**规则层（L1-L4）**：

1. **L1 来源锚定**（weight 0.35）：jieba 分词后逐句检查 token 重叠率，每句能否追溯到某篇检索源。阈值 0.15。
2. **L2 数值一致**（weight 0.25）：提取 answer 中的「数字+单位」对，交叉比对 source 中是否有相同数字。防止编造营收/增长等关键数据。
3. **L3 引用完整**（weight 0.20）：`[N]` 标记存在且 N 对应有效来源编号。防止“伪引用”。深度分析模式下放宽为检查文字引用（“分析”“评估”“显示”等）。
4. **L4 结构规范**（weight 0.20）：输出是否包含预期段落。RAG 模式期望 `# 摘要/要点/分析/风险` Markdown 标题；深度分析模式期望 `关键信号/影响分析/风险提示/后续关注` 括号段落。

**LLM 层（L5-L6）**：

5. **L5 LLM 质疑**：LLM 审查 answer + sources + L1-L4 检测结果，输出结构化 JSON 发现（严重程度 + 置信度）。发现规则层漏检的幻觉问题。
6. **L6 LLM 协助**：当 L1-L4 分数低于阈值时，LLM 主动修复问题（补充引用、修正无来源声明）。

**用户可见**：每层的得分、通过/未通过、未锚定的句子都会出现在最终输出的防幻觉报告中，不是藏在 metadata 里。

**集成点**：ScoringAgent 在 Phase 5 (Evolve) 调用 `check_hallucination` 工具，评分卡 + 防幻觉报告一起写入最终输出。

**架构演进**：最初是“凑数”的 6 层（完整性、事实核查与 L1 高度重叠）→ 重写为 4 层规则检查（每层有明确目标和独立分数）→ 增加 L5/L6 两个 LLM 层（规则层无法覆盖的语义级幻觉由 LLM 补充）。现在 6 层各有不可替代的职责。

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
| 防幻觉 | 六层透明校验：L1-L4 规则层（来源锚定 + 数值一致 + 引用完整 + 结构规范）+ L5 LLM 质疑 + L6 LLM 协助 |

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

**四层测试**（609 tests passing）：
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
| 同义词扩展 | 52 组双向同义词（内置 35 + 外部 JSON 17），如“英伟达” ↔ “NVIDIA” ↔ “NVDA”，O(1) 查找表 | 1.5 | 0ms |
| 概念关联 | 20 个行业概念单向关联（“芯片” → [半导体, 光刻, 晶圆]） | 0.6 | 0ms |
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

## Q24: 为什么要做 DictionaryRegistry？

**问题**：领域字典散落在 `dictionaries.py` 里，用 Python 常量定义（`STOCK_MAP = {...}`, `FINANCIAL_TERMS = {...}`）。每次要加新股票、新术语，就得改代码、重新部署。

**痛点**：
- **字典太弱**：STOCK_MAP 只有 11 个股票，jieba 分词词表只有 55 个词，金融术语只有 45 个
- **不可见**：字典覆盖率无法查看，哪里弱只能靠感觉
- **不可扩展**：想加 AI 领域术语就得改 `dictionaries.py`，每次提交一堆字典数据

**解决**：
- **中央注册中心**：`DictionaryRegistry` 统一管理 10 种字典（stock_map、financial_terms、synonym_lookup、jieba_words 等）
- **外部 JSON 热扩展**：把 JSON 文件放入 `data/dictionaries/` 目录，启动时自动加载并合并，不改源码
- **覆盖率可视化**：`reg.summary()` 一眼看出每种字典的规模，哪里弱补哪里
- **jieba 自动注入**：`set_jieba(jieba)` 一次性注入 91 个领域词，幂等（重复调用不重复注入）
- **向后兼容**：`dictionaries.py` 的模块级变量（`STOCK_MAP`、`SYNONYM_LOOKUP` 等）被 registry 自动替换为增强版本，现有 import 无需修改

**效果**：

| 字典 | 内置 | 增强后 | 增长 |
|------|------|--------|------|
| stock_map | 11 | 33 | +200% |
| financial_terms | 45 | 64 | +42% |
| industry_terms | 30 | 73 | +143% |
| synonym_lookup | 60 | 143 | +138% |
| jieba_words | 55 | 91 | +65% |

**设计原则**：字典数据属于配置而非代码。新增股票、术语、同义词只需要编辑 JSON，不应该触发代码变更。

---

## Q25: QueryPlanner 是怎么做的？为什么不做复杂的 DAG？

**问题**：现有流程是 QueryParser（纯规则）→ 检索 → LLM 回答。对于复杂查询（对比、时间线、深度分析），单次检索不够——需要先拆解为多个子查询，分别检索再合并。

**市面做法**：很多 RAG 系统用 DAG（有向无环图）做计划，子查询之间有依赖关系，支持顺序执行 + 结果传递。但 DAG 复杂度高，调试困难，对于我们的场景太重。

**我们的做法**：一次 LLM call，JSON 输出：
- **5 种意图**：factual / comparison / timeline / deep_dive / summary
- **每个子查询**带 source（kb/news/graph/all）和 mode（local/global/hybrid/mix）
- **执行策略**：parallel（并行检索）或 sequential（顺序检索）

**关键决策**：
- **不做 DAG**：单次 LLM call 生成的计划已经足够。子查询之间大部分是并行关系，不需要复杂依赖图
- **完全解耦**：`query_planner.py` 不改现有 `query_parser.py` 或 `retriever.py`，坐在新的一层
- **自动降级**：LLM 调用失败时回退为单子查询，不影响简单查询的响应速度
- **`is_simple` 属性**：单子查询跳过规划开销，零延迟

**效果**：

| 查询类型 | 示例 | 子查询数 | 策略 |
|----------|------|---------|------|
| factual | "矛台收盘价多少" | 1 | 直接检索 |
| comparison | "英伟达和华为芯片谁强" | 3 | 并行 |
| timeline | "OpenAI融资历程" | 3 | 顺序 |
| deep_dive | "商汤生成式AI前景" | 4 | 并行 |

**设计原则**：简单、可靠、不增加系统复杂度。如果将来需要子查询之间的依赖关系，可以在 QueryPlan 中加 `depends_on` 字段，不需要重写架构。

---

## Q26: LightRAG 图谱实验做了什么？

**目标**：验证 LightRAG SDK 能否从中文财经新闻中抽取知识图谱，支持跨文档的实体关系推理。

**踩坑**：

1. **API Key 优先级问题**：系统环境变量 `DASHSCOPE_API_KEY` 是旧的过期 key，但 dashscope SDK 优先读全局 `dashscope.api_key`，导致 `.env` 中的新 key 无效。解决：读 `.env` 文件优先，然后 `dashscope.api_key = API_KEY` 强制覆盖

2. **Tokenizer decode 返回空字符串**：LightRAG 用 `tiktoken`（需下载网络资源，国内不稳定），我们做了自定义 Tokenizer（字符索引，不需要下载）。但 decode() 返回空 → chunks 全空。根因：decode 必须用 min/max token 索引切片原始文本，而不是返回空

3. **Embedding 返回 Python list**：LightRAG 的 NanoVectorDB 用 `result.size` 属性，但 Python list 没有 `.size`。解决：`return np.array(embeddings, dtype=np.float32)`

**结果**（5 篇 AI 新闻）：
- 60 个实体、59 条关系、10 种实体类型
- 高连接度实体：Blackwell Ultra (14), SenseTime (10), OpenAI (9)
- 4 种查询模式（local/global/hybrid/mix）均能返回有意义的图谱增强答案

**状态**：已从独立 PoC 升级为主系统集成组件。摄取端通过 `ingest_router` 将 PDF/图片解析内容送入 LightRAG；查询端通过 `graph_tools.py` Function Calling 按需查询，由 QueryPlanner 的 `source: graph` 路由。

---

## Q27: 为什么做文档多模态解析？怎么设计的？

**问题**：原来的文件导入只支持纯文本（.txt / .csv）。但财报通常是 PDF，图片中也有重要信息（如架构图、表格截图）。纯文本导入丢失了这些结构化内容。

**设计**：
- **PDF 解析**：PyMuPDF 本地提取文本 + 表格，无需 API 调用，速度快
- **图片解析**：qwen-vl-plus 多模态模型，用结构化 prompt（`prompts.py` 的 `IMAGE_UNDERSTANDING_SYSTEM` + `IMAGE_UNDERSTANDING_PROMPT` + few-shot 正反例）
- **工具化**：`document_parse_tools.py` 用闭包注入模式（`_llm_ref` + `inject_document_parse_llm()`），对齐存量架构
- **双路径复用**：`IngestionAgent` 通过 `call_tool()` 委托，`ingest_router` 复用同一工具函数，保持 Agent 路径和 API 路径一致

**架构原则**：
- prompt 集中管理在 `prompts.py`，不散落在工具代码里
- 工具用闭包注入 + FunctionDef 注册，不直接依赖 LLM 实例
- Agent 只做编排（`call_tool`），解析逻辑在工具层

---

## Q28: LightRAG 图谱集成是怎么做的？为什么不每次都走图谱？

**问题**：LightRAG PoC 验证了知识图谱的可行性，但怎么集成到主系统是个问题。最初的做法是在 `_phase_index()` 里强制每次查询都走 LightRAG，导致：
- 图谱为空时白白浪费延迟
- 图谱和 BM25+Vector 评分体系不同，强制融合效果反而差
- 不符合 QueryPlanner 已经预留的 `source: graph` 路由设计

**重构后的设计**：

**摄取端（建图）**：
- PDF/图片解析后的文本在 `ingest_router` 中送入 `LightRAGAdapter.insert_texts()`
- 普通文本/新闻不走图谱，保持 BM25+Vector 通道
- 图谱存储在 JSON + GraphML 文件（`data/knowledge_base/lightrag/`），无需外部数据库

**查询端（问图）**：
- Agent 通过 Function Calling 按需调用 `query_knowledge_graph` / `get_graph_stats`
- QueryPlanner 根据查询意图决定是否路由到图谱（如实体关系推理类查询）
- 图谱结果返回 `retrieved_items` 对齐格式（`meta._source: graph`），Agent 可区分来源

**踩坑与修正**：
1. **async 桥接过度工程化**：最初用 `ThreadPoolExecutor` 绕过 event loop 死锁。实际上所有调用方（FastAPI sync handler、graph_tools）都在无 running loop 的线程中，直接 `asyncio.run()` 就行
2. **Pipeline 强制注入**：最初往 `PipelineScheduler` 里塞 `lightrag` 参数，每次查询都走图谱。违反“按需路由”原则，撤回
3. **死代码清理**：撤回 pipeline 注入后，`app_state.py` 里的 `scheduler.lightrag = adapter` 也变成了死代码，一并清理

**设计原则**：
- 图谱适合实体关系推理，不适合所有查询。QueryPlanner 路由 > Pipeline 硬编码
- async SDK 用 `asyncio.run()` 桥接，不用复杂的 loop 检测
- 闭包注入 + FunctionDef 注册，对齐其他工具的架构风格

---

## Q29: 前端为什么从单文件重构为模块化？

**问题**：原来前端是一个 920 行的 `app.js` + 一个 303 行的 `styles.css`，所有功能堆在一起。随着功能增加（查询、分析、聊天、KB 管理、导入、概览）变得越来越难维护。

**重构**：
- **JS 拆分**：按业务域拆为 9 个 ES Module（`api.js`、`ui.js`、`overview.js`、`ingest.js`、`kb.js`、`query.js`、`analyze.js`、`chat.js`、`render.js`），每个模块只负责一个领域
- **CSS 拆分**：按功能拆为 6 个文件（`base.css` 变量/重置、`components.css` 通用组件、`overview.css`/`ingest.css`/`query.css`/`chat.css` 各面板专属）
- **全局注册**：`app.js` 只做 import + `window._xxx = func` 全局注册，HTML onclick 保持不变

**收益**：
- 单文件从 920 行降到每个模块 100-300 行
- 新增功能只需新建或修改对应模块，不会误伤其他功能
- CSS 变量集中在 `base.css`，改主题色只改一处

**设计原则**：原生 JS ES Module，不引入 React/Vue 等框架。项目是 RAG 后端系统，前端是轻量级展示层，不需要重框架。

---

## Q30: 可折叠卡片是怎么设计的？

**问题**：导入数据页有“文件导入”“新闻抓取”“知识库管理”多个区域，全部展开时页面很长，用户要滚动才能找到需要的功能。

**设计**：
- HTML 卡片加 `data-collapsible` 属性，`ui.js` 的 `initCollapsibleCards()` 统一初始化
- 点击卡片标题栏切换折叠状态，CSS transition 动画平滑过渡
- 每个卡片独立折叠/展开，互不影响

**收益**：用户可以根据当前任务只展开需要的区域，减少视觉干扰。首次访问时关键区域默认展开、次要区域折叠。

---

## Q31: 为什么做文件上传？和目录导入有什么区别？

**问题**：原来的导入只支持“输入服务器本地目录路径”，系统扫描目录下的文件。这有两个问题：
1. 用户必须先把文件放到服务器的特定目录
2. PDF 和图片虽然已经支持解析，但前端没有直接入口

**设计**：
- **前端**：新增「文件上传」卡片，拖拽区 + `<input type="file" accept=".pdf,.png,.jpg,.jpeg,.webp">`，支持多文件选择
- **后端**：`POST /api/ingest/upload`，`multipart/form-data` 格式，用 `python-multipart` 依赖
- **复用已有解析**：服务端复用 `_parse_pdf_text()` / `_describe_image()`，解析后走相同的入库流程（KB + LightRAG 图谱）
- **临时文件**：上传文件保存到 `tempfile.mkdtemp()`，处理完成后 `shutil.rmtree()` 清理

**两种入口的定位**：
- **目录导入**：批量导入大量文件（几十到几百篇），适合初始化知识库
- **文件上传**：快速导入少量文件（1-5 篇），适合临时分析

---

## Q32: 防幻觉 Guard 在深度分析下为什么得分很低？怎么修的？

**问题**：新闻解读 / 话题调研的防幻觉评分总是很低（可信度 22%，L3 引用完整 20%，L4 结构规范 0%）。

**根因**：HallucinationGuard 的 L3/L4 规则层是为 RAG 查询设计的：
- **L3** 期望答案中有 `[1]`、`[2]` 等引用标记 → 深度分析输出没有这种标记 → fallback 到 0.2
- **L4** 期望 `# 摘要`、`# 要点`、`# 分析`、`# 风险` 等 Markdown 标题 → 深度分析用的是 `【关键信号】【影响分析】【风险提示】【后续关注】` → 0% 匹配

**修复**：给 Guard 加 `mode` 参数（`"rag"` vs `"analysis"`），深度分析时自动切换评分标准：

| 层 | RAG 模式 | Analysis 模式 |
|---|---|---|
| L3 | 期望 `[N]` 引用，无引用 → 0.2 | 检查文字引用（“分析”“评估”等），有引用 → 0.8，无引用 → 0.6 |
| L4 | 期望 `# Markdown` 标题 | 期望 `关键信号/影响分析/风险提示/后续关注` 括号段落 |

**效果**：L3 从 20% → 80%，L4 从 0% → 100%，整体可信度从 22% 提升到 60-70%。RAG 查询路径不受影响（默认 `mode="rag"`）。

**设计原则**：同一套 Guard 代码，通过 `mode` 参数感知上下文，不同场景用不同评分标准。而不是写两套 Guard。

---

## Q33: 知识库数据质量为什么差？怎么修的？

**问题**：知识库平均文档长度只有 133 字，22% 的文档不足 100 字，19 组近似重复（每组 3-5 条），还有 pytest fixture 残留数据（"doc from file A"）。检索质量极差，防幻觉评分也很低。

**发现方式**：先清洗数据（137 → 81 篇），然后加载真实新闻后测试，发现检索和可信度仍然不理想。

**根因分析**：
- **新闻只存元数据**：`/api/ingest/news` 把新闻存到 `news_metadata.json`（只有标题/来源/时间），内容从未进入 KB
- **内容被截断**：`rss_fetcher.py` 三个 API 都对 content 做 `[:500]` 截断
- **文件导入裸奔**：文件直接存入 KB，没有经过预处理流水线

**修复**：
- **移除截断**：`rss_fetcher.py` 和 `news_tools.py` 中所有 `content[:500]` 全部移除，保留完整内容
- **新闻自动入 KB**：`/api/ingest/news` 新增自动入 KB 逻辑，经过 `_preprocess_docs()` 流水线（清洗 + 相关性门控 + 最小长度过滤 + 分类）+ 去重
- **文件导入加质量门控**：文件导入和上传都走同一预处理流水线，不达标的文档被拒绝
- **Archive 格式升级**：从纯文本拼接改为结构化存储（title/content/text/metadata 分字段）

**教训**：数据质量是系统效果的基础。算法再好，如果 KB 里全是垃圾，检索和防幻觉都不可能好。“先治数据，再治算法”。

---

## Q34: 防幻觉可信度为什么总是很低（37%、46%）？

**问题**：新闻解读的可信度一直很低（37%-46%），L4 结构规范 0%，L5/L6 也是 0%。

**发现方式**：用户反复测试新闻解读，每次都看到高风险警告。

**三个根因**：
1. **`_compute_overall` 权重未归一化**：L5+L6 权重占 40%，被跳过时（LLM 未触发）满分只有 60%，怎么算都是高风险
2. **L4 正则太严格**：分析模式下找 `关键信号`/`影响分析` 等精确词，但 LLM 输出是 `核心信号`/`主要信号`/`利好`/`利空` 等变体
3. **Guard 检查的文本和前端显示的内容脱节**：Guard 检查 LLM JSON 输出的 `analysis` 字段（一段短文本），但 L4 要找的 `关键信号`/`影响分析`/`风险提示`/`后续关注` 在 structured 的独立字段里

**修复**：
- **权重归一化**：`_compute_overall()` 只对实际执行的层求加权平均，跳过的层不参与计算
- **L4 放宽匹配**：`_EXPECTED_SECTIONS_ANALYSIS` 扩展为包含多种变体（核心/主要/重要信号、冲击/波及/利好/利空、潜在风险/下行风险等）
- **`_build_full_analysis_text()`**：在 `analysis_router.py` 新增函数，从 structured 输出重建完整文本（关键信号 + 影响分析 + 综合分析 + 风险提示 + 后续关注），和用户在屏幕上看到的内容一致

**教训**：
- 评分规则必须和实际输出格式对齐。检查的文本和用户看到的不同，评分就没有意义
- 权重归一化是基础设计问题，不是调参问题
- 调试评分问题时，先看“尺子”有没有问题，再看“被量的布”

---

## Q35: 启动日志为什么太哆？怎么清理的？

**问题**：启动时输出 30+ 行 INFO 日志，包括 Chroma dedup、KV load、Role LLM、Embedding cache、DictionaryRegistry、Coordinate 注册等内部细节，调试时噪音太大。

**根因**：所有日志都是 INFO 级别，没有区分“用户关心的启动状态”和“实现细节”。

**修复**：
- **分级**：Chroma/dedup/KV load/Embedding cache/DictionaryRegistry/校验和/LightRAG init/会话加载/Stats 等全部降为 DEBUG
- **抑制第三方库**：`lightrag`、`nano-vectordb`、`chromadb`、`httpx`、`openai` 的 logger 设为 WARNING
- **精简 KB 日志**：来源分布字典不再全部打印，改为显示来源数量
- **合并启动日志**：只保留 KB 摘要（篇数/索引状态/元数据数）+ 启动完成

**效果**：从 30+ 行 → 仅 2 行 INFO（KB 状态 + 启动完成）。

**原则**：启动日志应该回答“系统准备好了吗？”，而不是“系统在做什么？”。

---

## Q36: 深度分析为什么要从“直接调 service”改为走 Agent 链？

**问题**：Q8 修复了深度断裂后，新闻解读和话题调研已经能输出结构化分析了，但走的是 `analysis_router` 直接调 `services/analysis.py`——防幻觉评分和全链路打分完全没接入。用户看到的分析结果没有任何质量评估。

**发现方式**：对比同一个功能的两条路径：Pipeline 走 Agent 链有完整的 ScoreCard + Guard，而深度分析页签直接调 service 什么评分都没有——**同一个功能，两条路径质量差异巨大**。

**修复**：
- `analysis_router` 新增 `_run_analysis_via_agent_chain()`：设置 `orchestrator.set_pipeline(["AnalysisAgent", "ScoringAgent"])` 走完整 Agent 链
- `AnalysisAgent` 新增 `_build_scoring_text()` 方法：重建完整分析文本（关键信号 + 影响分析 + 综合分析 + 风险提示），供 Guard 检查
- `AnalysisAgent` 在 metadata 中塞入三个字段：`scoring_source_items`、`scoring_mode="analysis"`、`scoring_text`
- `ScoringAgent` 读取 metadata 三个通用字段，不需要知道上游是什么功能
- **Fallback**：Agent 链失败时自动降级回直接 service 调用，保证功能可用性

**收益**：深度分析现在也有防幻觉评分 + 可信度报告，和 Pipeline 路径质量对齐。

---

## Q37: ScoringAgent 为什么要做成“通用公共能力”？

**问题**：ScoringAgent 原本是为 Pipeline 设计的，检查的文本是 `context.final_answer`，source 是 `context.intermediate_findings`。深度分析走 Agent 链后，这两个字段的数据结构和 Pipeline 场景完全不同。

**决策**：不是写两套 ScoringAgent，而是设计一个**通用接口契约**：

```python
# 任何 feature 往 metadata 塞这 3 个字段即可接入评分：
metadata["scoring_source_items"] = [...]   # 检索源，供 L1 来源锚定
metadata["scoring_mode"] = "analysis"       # "rag" 或 "analysis"，控制 L3/L4 标准
metadata["scoring_text"] = "..."           # 待检查文本（如果不用 final_answer）
```

ScoringAgent 内部逻辑：先读通用字段（优先），再 fallback 到 RAG 场景字段。这样新增任何 feature（如 Pipeline、深度分析、未来的事件分析）都只需设置 3 个 metadata 字段，不需要改 ScoringAgent 代码。

**设计原则**：Agent 间通过 metadata 字段约定通信，而非硬编码依赖。新增上游功能 + 3 个字段 = 接入评分，ScoringAgent 零修改。

---

## Q38: 智能查询为什么加双模式？

**问题**：后端有 5 阶段 Pipeline（Fetch → Index → Process → Output → Evolve），是系统最强的查询能力，但前端只暴露了一个简单的 KB 搜索框。用户完全不知道后面有这些能力，反馈“做这么厚实，我都不知道怎么用”。

**决策**：不新增标签页，在现有搜索框加模式切换：
- **知识库问答**：混合检索 + LLM 回答（简单快速）
- **深度调研**：完整 5 阶段 Pipeline（抓新闻 + 入 KB + AI 分析 + 报告 + 核查）

**用户友好化**：所有技术术语翻译为用户语言：
- “KB 搜索” → “知识库问答”
- “Pipeline” → “深度调研”
- “Fetch/Index/Process/Output/Evolve” → “抓新闻/入知识库/AI分析/生成报告/事实核查”
- “槽位” → “完成度”
- 每个模式加步骤提示（①②③），用户一看就知道先干嘛后干嘛

**教训**：能力做得再好，如果用户看不到、看不懂，等于没做。前端文案和后端架构同样重要。

--

## Q39: 检索模块为什么做结构重组？

**问题**：`retriever.py` 这个名字只体现了"检索"，但它实际干两件事：入库调度（chunk → embed → Chroma）和检索调度（parse → BM25+Vector → fuse → rerank → gate）。更致命的是，检索器的装配逻辑藏在 `core/factory.py` 里，chunker 写好了但从未被接入运行路径——跑了这么久等于白写。

**发现方式**：对照产品设计文档审计检索能力时，发现 chunker 代码存在但 `HybridRetriever.__init__` 里 `chunker=None`。factory 没传。

**修复**：
- `retriever.py` → `hybrid_engine.py`：文件名准确反映入库+检索双调度职责
- 新建 `retrievers/factory.py`：检索器唯一装配入口，从 `core/factory.py` 迁出。打开这个文件就能看到所有零件和配置
- 新建 `retrievers/README.md`：文件级职责说明 + 两条链路的执行顺序
- `core/factory.py` 保留 re-export 兼容旧 import

**设计原则**：一个目录 = 一个能力域的全部。打开 `retrievers/` 就能看到检索系统的装配图、调度中心、所有零件。不用跨 4 个文件猜谜语。

---

## Q40: Chunker 重构解决了什么问题？

**问题**：原来的 chunker 是固定 1500 字硬切，不管文档类型。一篇 300 字的短新闻被切成 1 个 chunk 还好，但 5000 字的研报按固定长度切，段落被从中间劈开，语义完整性被破坏。

**设计**：
- **短文不切**：`skip_threshold=500`，<500 字的文档直接作为单个 chunk，保持原文完整性
- **长文按段落切**：递归切分器优先在段落边界（`\n\n`）断开，不在句子中间劈
- **按文档类型路由**：`DOCTYPE_STRATEGY` 映射表，不同 doc_type 用不同 chunk_size/overlap

| doc_type | chunk_size | overlap | 理由 |
|----------|-----------|---------|------|
| news | 500 | 0 | 新闻短，不切或只切一刀 |
| financial_report | 1500 | 100 | 财报长，按段落切 |
| research | 1500 | 100 | 研报同上 |
| other | 1000 | 80 | 默认策略 |

**关键修复**：chunker 之前从未被接入 `HybridRetriever`（factory 没传），这次在 `retrievers/factory.py` 中显式组装。

---

## Q41: 检索质量门控是怎么做的？

**问题**：用户问一个知识库里根本没有的问题（如"比特币价格"），系统仍然返回一堆不相关的结果，LLM 基于垃圾上下文编造答案。防幻觉 Guard 能事后检测，但为什么不从一开始就拦住？

**设计**：在 `hybrid_engine.py` 的 `search()` 末尾加硬拦截：
- Rerank 后 top1 分数 < 0.15 → 直接返回空结果 + 诊断信息
- 只在有 Rerank 时生效（RRF 分数量纲 0~0.02，Rerank 量纲 0~1，不能混用）
- 拦截信息存在 `last_gate_info` 字段，包含 blocked/stage/reason/query/top_score/threshold

**诊断透出**：`tools/core.py` 的 search tool 检测到拦截后返回：
```json
{"gate_blocked": true, "gate_info": {"stage": "quality_gate", "reason": "...", "top_score": 0.08}}
```
Agent 和前端都能看到"为什么没结果"，方便持续调优阈值。

**设计原则**：宁可告诉用户"没找到"，也不给垃圾结果让 LLM 编。门控是防幻觉的第一道防线。

---

## Q42: Metadata 正则抽取是怎么设计的？为什么不用 LLM？

**问题**：文档入库时需要抽取 company/publish_date/sector 等元数据，用于后续检索过滤和来源展示。

**决策**：用正则 + 词典，不用 LLM：
- **成本**：每篇文档调一次 LLM 抽取 metadata，几百篇文档导入时 API 费用和时间都不可接受
- **确定性**：正则抽取结果稳定可复现，LLM 每次可能返回不同格式
- **速度**：正则 <1ms，LLM ~3s

**实现**（`retrievers/metadata.py`）：
- **company**：词典匹配（STOCK_MAP + stocks_extended.json 33 条）→ 正则兜底（XX科技/集团/股份）
- **publish_date**：多模式正则（年月日/季度/ISO格式），按优先级排列
- **sector**：关键词→行业映射，返回出现最多的行业

**数据流**：`ingest_router._preprocess_docs()` 第 4 步调用 `extract_metadata()` → 写入 `doc["meta"]` → `_flatten_meta()` 按白名单过滤后存入 ChromaDB。

---

## Q43: synthesize_report JSON 截断是怎么排查的？

**问题**：导入文章后 `synthesize_report` 始终报 "JSON 解析失败"，3 次重试全挂，走了兜底逻辑。

**排查过程**：
1. 第一反应是 `max_tokens=2048` 太小 → 改为 4096 → **仍然失败**
2. 加诊断日志：打印响应长度、开头、结尾、括号平衡状态
3. 发现：改了 4096 后第一次仍然失败，是因为 **LLM 响应缓存**——第一次失败的截断响应被缓存了，重试拿到的还是同一份烂数据
4. 清除缓存后，4096 生效，响应 1209 字，一次解析成功

**根因**：两个问题叠加：
- `max_tokens=2048` 确实太小（DashScope 客户端默认 4096，report_tools 自己限死了）
- `call_json()` 的 `use_cache=True` 在第一次失败后缓存了截断响应，后续重试拿缓存 → 永远失败

**修复**：max_tokens 2048 → 4096 + 清缓存。重试时 `use_cache=(attempt == 0)` 已经设计了（重试不缓存），但第一次的缓存已经写入。

**教训**：
- 缓存是双刃剑：成功时省 API 调用，失败时毒化后续重试
- 调试 JSON 解析问题时，先加日志看"响应到底长什么样"，比猜代码逻辑快 10 倍
- 改配置后如果"没生效"，先怀疑缓存，再怀疑代码

---

## Q44: 防幻觉诊断信息为什么用户看不到？怎么修的？

**问题**：产品设计文档要求"让用户更清楚知道哪些地方检测了，且检测效果如何"，但实际上用户只能看到一个百分比分数，不知道具体哪句话有问题、哪个数字不匹配、L5/L6 为什么没跑。

**发现方式**：对照产品设计文档（`产品设计文档.txt`）逐条审计 API 响应，发现后端 Guard 产出了丰富的诊断数据（未锚定句子、不匹配数字、逐句判定），但 API 层全部截断成只剩 `{"score": 0.75}`。

**三个根因**：
1. **API 层截断**：`kb_router.py` 和 `analysis_router.py` 的 3 个构造点都用 `{k: {"score": v.get("score")} for ...}` 只取 score，丢弃所有诊断字段
2. **L5/L6 静默省略**：规则层 ≥85% 或无 LLM 时，L5/L6 直接不出现在 `checks` 字典里，前端完全无感知
3. **3 个构造点重复代码**：kb_router、analysis_router fallback、agent chain 各写一遍相同的截断逻辑，改一处漏两处

**修复**：
- **`guard/reflector.py`**：L5/L6 跳过时写入 `{"skipped": True, "skip_reason": "LLM 未注入" / "规则层已通过"}`，禁止静默省略；`_compute_overall()` 排除跳过层；`format_report()` 跳过层显示"未执行 — 原因"
- **`guard/serializer.py`（新建）**：共享序列化 helper，白名单机制透出每层诊断详情（L1 的 unanchored 句子、L2 的 unmatched 数字、L6 的 per_sentence 逐句判定），过滤 raw 字段防止泄漏 LLM 原始输出
- **3 个 API 调用点**：统一替换为 `serialize_guard_result(guard_result)`，一处定义、三处复用

**效果（API 响应对比）**：
```
之前: "L1_source_grounding": {"score": 0.75}
现在: "L1_source_grounding": {"score": 0.75, "anchored": 3, "total": 4,
       "unanchored": ["公司预计2025年将实现盈利，行业龙头地位进一步巩固"]}

之前: L5/L6 不出现
现在: "L5_llm_critique": {"skipped": true, "skip_reason": "LLM 未注入，LLM 层无法执行"}
```

**教训**：
- 后端产出了数据 ≠ 用户看到了数据。API 序列化层是信息透出的咽喉，截断逻辑必须审计
- 多处重复的构造逻辑必须抽成共享函数，否则改一处漏两处
- "不接受静默少测"是产品硬约束——跳过可以，但必须告诉用户为什么跳过

---

## Q45: BM25 为什么从 rank_bm25 换成 SQLite FTS5？

**问题**：原 BM25 用 `rank_bm25.BM25Okapi` 纯内存实现，每次 `add()` 都全量重建整个语料（O(N)）。81 篇时无感，但文档量增长后每次导入一个文件就重建一次，且重启后索引丢失需从头构建。面试时一说"BM25 每次全量 rebuild"立刻被 diss 为 demo 级。

**对比成熟项目**：Dify 用 Elasticsearch / Weaviate 等支持增量写入的引擎；RAGFlow 用 ES 倒排索引。共同点：**写入即生效，无需重建**。

**为什么选 SQLite FTS5 而非 ES**：
- ES 需要 JVM + 独立服务 + IK 插件，本地开发体验极差
- 项目规模（百~千篇级）远未到 ES 的设计目标（亿级）
- FTS5 是 Python 标准库 `sqlite3` 内置，零依赖、零部署
- 天然支持 BM25 排序（`bm25()` 函数）、增量 INSERT/DELETE、持久化

**实现**：
- `bm25_engine.py` 全部重写：建 `kb_fts` FTS5 虚拟表，文档经 jieba 分词后空格拼接存入
- 搜索：分词 → FTS5 MATCH（OR 语义）→ `bm25()` 排序
- `hybrid_engine.add()` → `bm25.add(documents)`：真增量 INSERT
- `hybrid_engine.remove_by_indices()` → `bm25.remove_by_docs()`：真增量 DELETE
- 索引落盘 `data/knowledge_base/bm25_index.db`，重启秒加载
- `requirements.txt` 移除 `rank_bm25`

**效果**：
```
之前: add 50篇 → 全量 rebuild × 50次（每次 O(N)）
现在: add 50篇 → 50次 INSERT（每次 O(1)）
之前: 重启 → 索引丢失，需重新 build
现在: 重启 → 打开 .db 文件，毫秒级恢复
```

---

## Q46: Embedding 并发化怎么做的？提速多少？

**问题**：`DashScopeEmbedding.embed()` 内部串行循环调 API（每批 10 条，等返回再发下一批）。200 个 chunk = 20 批 × ~1s/批 = **串行 20s**。`embedding_cache.py` 外面还套了一层手动切 10 的冗余循环。

**成熟做法**（Dify / 生产 RAG）：asyncio + Semaphore(5~10) 并发发 batch，吞吐提升 5-10x。

**实现**（最小改动）：
- `dashscope_client.py`：`embed()` 拆出 `_call_batch()` 单批方法，多批时用 `ThreadPoolExecutor.map()` 并发执行（MAX_WORKERS=5），`pool.map` 保证结果顺序
- 单批（≤10 条）直接调用，无线程池开销
- `embedding_cache.py`：去掉冗余的 `for j in range(0, len, 10)` 循环，一次性把未命中文本交给 `embedder.embed_documents()`（内部已会分批+并发）

**为什么用线程池而非 asyncio**：DashScope SDK 是同步阻塞的（`dashscope.TextEmbedding.call()`），没有 async 接口。线程池是最干净的并发方式，无需改写 SDK 调用方式。

**效果**：
```
之前: 200条 → 20批串行 → ~20s
现在: 200条 → 20批 / 5并发 = 4轮 → ~4s（5x 提速）
```

**教训**：
- `ThreadPoolExecutor` 早就 import 了但从没用过——"import 了 ≠ 用了"
- 两层串行叠加（client 内 + cache 外）要一起修，否则只修一层等于没修

---

## 关键数字（面试时可引用）

| 指标 | 数据 |
|---|---|
| 总 commit 数 | ~65 |
| Agent 数量 | 4（从 7 精简） |
| 注册工具 | 32 个，跨 11 个模块 |
| 测试覆盖 | 626 tests（28 个测试文件） |
| 知识库文档 | ~500+ 篇（去重后） |
| 检索延迟 | BM25 FTS5 < 10ms（SQLite 持久化），ChromaDB ANN < 200ms |
| 查询规划 | QueryPlanner: 5 种意图 + 来源/模式感知子查询，LLM 失败自动降级 |
| 查询扩展 | 52 组同义词 + 20 个概念关联，DictionaryRegistry 统一管理，规则层 0ms + LLM 层可选 |
| 知识图谱 | LightRAG 集成: PDF/图片 → 实体关系抽取 → Function Calling 查询（local/global/hybrid/mix） |
| 文档解析 | PyMuPDF (PDF, 本地) + qwen-vl-plus (图片多模态)，闭包注入 + FunctionDef 注册 |
| 检索架构 | `retrievers/` 目录自包含：factory(装配) + hybrid_engine(调度) + 12 个子模块，README 文件级说明 |
| 入库性能 | BM25 真增量（SQLite FTS5）+ Embedding 5 路并发（ThreadPoolExecutor），200 chunk embedding ~4s |
| 检索门控 | Rerank score < 0.15 硬拦截 + `last_gate_info` 全量诊断（stage/reason/top_score/threshold） |
| Chunker | skip_threshold=500 + DOCTYPE_STRATEGY 按文档类型路由（news/report/research/other） |
| Metadata | 正则抽取 company/publish_date/sector，CHROMA_META_WHITELIST 白名单过滤，INPUT 侧闭环 |
| 知识库导入 | 预处理门控（清洗 + 相关性 + 长度 + 分类 + 去重 + metadata 正则抽取），新闻自动入 KB |
| 全链路评分 | 6 层防幻觉（4 规则 + 2 LLM）+ 5 阶段打分，**双模式** + **权重归一化** + **诊断数据完整透出**（`guard/serializer.py` 白名单序列化，L5/L6 跳过显式标记） |
| API 架构 | 4 个 FastAPI Router（KB / Ingest / Analysis / Query），Ingest 支持目录导入 + 文件上传 |
| 意图路由 | 4 种意图 + general 兜底（kline / event_impact / report / news / general），深度分析页签额外有 `deep_topic` |
| 领域字典 | 10 种字典类型，外部 JSON 热扩展（`data/dictionaries/*.json`），STOCK_MAP 33 条 / synonym 143 条 |
| 前端架构 | 原生 JS ES Module（9 模块） + 6 分层 CSS + 可折叠卡片（`data-collapsible`） |
| 智能查询 | 双模式（知识库问答 / 深度调研），用户友好文案，步骤提示指引 |
| 深度分析 | Agent 链 (AnalysisAgent → ScoringAgent)，ScoringAgent 通用接口契约（3 个 metadata 字段） |
