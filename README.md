


查询 → 意图路由 → Agent 链 → 混合检索 → 经防幻觉护栏校验的有据回答。专为中国市场的 AI 板块打造（股票、新闻、研报）。

> ### ⚠️ 诚实声明：V1.0，正在积极进行 TDD 加固
>
> V1.0 是一个**能用但粗糙**的产品。它快速铺开了很大的功能面——多 Agent 路由、混合检索、图谱 RAG、Web UI——但**质量是靠文字断言的，而非用测试度量的**。RAG 流水线基本*没有*端到端验证。
>
> **当前使命是 TDD。** 我们正在补建一套测试护栏，**一步一个脚印**地走完 RAG 流水线：推导每一步的契约、用真实行为去验证、修复暴露的问题、并把修复钉成永久的回归资产。不做大爆炸式重写。本 README 反映这一真实状态，而非过度包装。

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-273%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-gray)

---

## 当前使命 —— 护栏驱动的加固

**方法（对 RAG 流水线的每一步重复执行）：**

1. **从代码契约推导"应然"** —— 是规格，不是猜出来的答案。
2. **用一次性探针脚本对真实组件观测"实然"**。
3. **对照。** 应然与实然分歧之处 → 是真实缺陷，不是假想问题。
4. **就地修复**，然后**把探针固化为永久测试。**

**每条测试必须满足的四条标准：**

| 标准 | 含义 |
|----------|---------|
| ① 证明效果 | 断言行为/结果，而不只是形状或"没崩" |
| ② 可观测 | 被测对象会发出可追踪 / 可断言的输出 |
| ③ 一致性 | 同输入 → 同输出；确定性且离线 |
| ④ 回归锁定 | 已修复的 bug 被钉死，无法悄悄回退 |

**目前的流水线覆盖：**

| # | RAG 步骤 | 状态 | 测试资产 |
|---|----------|--------|-----------|
| 0 | 输入清洗 —— 预处理 | ✅ 已验证 | `tests/test_preprocess_step0.py` (39) |
| 1 | 入库 —— 切分 + 索引 | ✅ 已验证 · 发现并修复 1 个 bug | `tests/test_ingest_step1.py` (15) |
| 2 | 查询解析 | ✅ 已验证 · 发现并修复 2 个 bug | `tests/test_query_parser_step2.py` (30) |
| 3 | 召回 —— BM25 + 向量 | ✅ 已验证 · 发现并修复 1 个 bug | `tests/test_recall_step3.py` (24) |
| 4 | RRF 融合 | ✅ 已验证 · 扭负增值为正（3 处根因修复） | `tests/test_fusion_step4.py` (9) |
| 5 | 过滤 / 质量门控 | ✅ 已验证 · 发现并修复 1 个 bug | `tests/test_filter_gate_step5.py` (20) |
| 6 | 生成有据化 | ✅ 已验证 · 发现并修复 2 个 bug | `tests/test_grounded_generation_step7.py` (22) |

**流水线 Step 0-7 已全部闭环。** 另有上下文组装层 `tests/test_context_builder.py` (9)，输出层评估 `tests/test_output_metrics.py` (13) + `tests/test_output_eval_golden.py` (20)，可观测性层 `tests/test_observability_tracing.py` (14)。

**新增 L1 检索评估层 —— 离线 IR 指标 + 分通道对比 + 基线回归锁：**

| 测试资产 | 数量 | 作用 |
|-----------|------|------|
| `tests/test_eval_metrics.py` | 26 | IR 指标实现对照手算（recall / precision / MRR / NDCG / MAP） |
| `tests/test_eval_golden.py` | 13 | golden qrels 语料结构（分级相关性、显式负例、场景标签） |
| `tests/test_eval_retrieval.py` | 9 | 评估闭环：离线确定性、分通道可见、融合核心指标无回归 |

**合计：273 通过 —— 离线运行，无需 API key。** 测试通过 `conftest` fixture 强制走本地检索（BM25 + Jaccard），因此确定性、无网络依赖。LLM/embedding 仅在测试显式调用时才保持真实；外部数据 API 一律 mock。

**护栏已抓到的真实 bug（每个都有回归测试钉死）：**

> - **Step 2 —— 日期被误判为股票代码。** `_extract_stock` 的 6 位数字正则抓取了 8 位日期（`YYYYMMDD`）的前 6 位，导致 `"20240601"` 被解析成 `stock_code="202406"`。已用数字边界 `(?<!\d)(\d{6})(?!\d)` 修复。
> - **Step 3 —— BM25 chunk 碰撞。** 一篇文档切分后的所有 chunk 继承同一个 `meta.doc_id`，导致 FTS5 去重只悄悄保留最后一个 chunk（存活率 1/3）。已用复合 key `{doc_id}_c{chunk_id}` 修复。
> - **Step 1 —— 删除索引错位。** `kb_router` 的删除端点把原始文档列表的位置索引传给 `retriever.remove()`，但检索器持有的是*切分后*的列表——一旦长文被切分，位置就会错位，删掉错误的 chunk。已用 `remove_by_doc_ids()` 修复。
> - **Step 2 —— 核心词被静默丢弃。** 关键词抽取的兜底分词是"全有或全无"：只要命中任意一个词典词，未登录的意图/实体词（`融资`、`昇腾`、`投资`、`财报`）就永远不进分词，饿死 BM25。已改为始终补充残余（未覆盖）token——在 BM25 通道上可度量地把融资文档召回从 MISS 翻转为 HIT。
> - **Step 5 —— 日期过滤静默失效。** `query_parser` 产出无 `$` 前缀的 `{"gte":...}`，而 `apply_filters` 只认 `$gte`/`$lte`，未知操作符被忽略 → 放行全部文档。已用 op 键规范化修复。
> - **生成有据化 —— L1 锚定阈值过宽，编造事实 0.964 分/risk=low 放行。** 编造句仅凭领域共享词（公司/收购/交易/2024）overlap 0.182 ≥ 旧阈值 0.15 即"锚定成功"。分离度实测（编造 ≤0.25 < 合法改写 ≥0.50）后阈值提到 0.30，干净分离且不误杀合法改写。
> - **生成有据化 —— 引用标记 `[N]` 被 L2 当成裸数字。** `extract_numbers` 把 `[1]`/`[3]` 提取为数字，L3 要求的规范引用反而污染 L2（满分回答被扣到 0.85）。已在提取前剥离 `[N]`。

---

## 功能清单（V1.0 特性 —— 已构建；验证进行中）

这些都已存在于代码中。它们**尚未**全部被护栏覆盖——补齐这一缺口正是上面进行中的工作。

- **意图路由。** `AgentRouter` 把查询分类为某种意图，并组装匹配的 Agent 链（Ingestion / Analysis / Scoring）。无需手动选择流水线。
- **瘦 Agent，胖工具。** Agent 只通过 `self.call_tool()` 编排；业务逻辑住在 11 个工具模块的已注册工具里。由架构守卫规则强制执行。
- **混合检索。** BM25 + ChromaDB 向量检索 + RRF 融合 + rerank + metadata 过滤 + 查询扩展。*检索质量正是护栏要度量的目标——目前尚未验证完毕。*
- **离线检索评估。** `financial_rag/eval/` 是一套离线、确定的检索质量 harness：golden qrels（分级相关性 0-3）+ 标准 IR 指标（recall / precision / MRR / NDCG / MAP）+ bm25 / vector / fusion 分通道对比 + `baseline.json` 回归锁。已用它量化定位到融合阶段的检索弱点（单通道满分、RRF 融合后归零）。
- **查询规划。** `QueryPlanner` 用一次 LLM 调用把复杂查询拆解为带 source/mode 的子查询；简单查询跳过它。
- **防幻觉护栏。** ScoringAgent 在返回结果前运行多层护栏（规则层：来源有据、数值保真、引用完整、结构合规；外加 LLM 批判/辅助）。
- **可观测的上下文组装。** 单一的 `ContextBuilder` 把检索到的 chunk 转成同时喂给 LLM 和护栏的 grounding，并发出保真轨迹（逐条 kept/dropped 及原因）。已被测试覆盖。
- **图谱 RAG。** LightRAG 知识图谱接入入库流程；通过 function-calling 工具按需查询，由 `QueryPlanner` 的 `source: graph` 路由。
- **多模态解析。** PDF 走 PyMuPDF（本地，无 API），图片走 qwen-vl-plus。
- **Web UI + CLI。** FastAPI 暗色主题 UI（拖拽上传文件、知识库管理、双模式查询）以及一个 CLI。
- **可扩展词典。** `DictionaryRegistry` 合并来自 `data/dictionaries/` 的外部 JSON——覆盖面无需改代码即可增长。
- **弹性 LLM 层。** `LLMCaller` 为每次调用包裹重试、JSON 修复和响应缓存。

---

## 架构

```
用户层 —— FastAPI Web UI（暗色主题）+ CLI
   │
调度层 —— AgentRouter（意图 → Agent 链） · PipelineScheduler（5 阶段执行）
   │
查询规划 —— QueryPlanner（LLM 拆解复杂查询 → 带 source + mode 的子查询）
   │
Agent 层（3 个）
   ├─ Ingestion  抓取并索引源文档
   ├─ Analysis   按意图路由到工具、组装结果
   └─ Scoring    防幻觉护栏 + 质量评估
   │
   ├─ 检索        BM25 + Chroma · RRF + Rerank · +Graph
   ├─ 数据 API    Tushare · 10jqka/东方财富 · LightRAG（已集成）
   └─ LLM 层      LLMCaller（重试+缓存+JSON） · ModelRouter（4 档，Qwen）
```

---

## 路由如何工作

AgentRouter 把每个查询分类为某种意图，然后调度器组装 Agent 链：

| 意图 | 示例查询 | 链 |
|--------|---------------|-------|
| K 线分析 | "看下 600519 的 MACD" | Analysis → Scoring |
| 事件影响 | "这次收购是利好还是利空？" | Analysis → Scoring |
| 研报综合 | "总结一下 Q4 财报" | Ingestion → Analysis → Scoring |
| 新闻深读 | "这周 AI 圈发生了什么？" | Analysis (deep) → Scoring |
| 话题调研 | "调研一下 AI 芯片赛道" | Analysis (deep_topic) → Scoring |
| 通用 | 其他任意查询 | Ingestion → Analysis → Scoring |

用户输入一个自然语言问题；剩下的由系统决定。

---

## 技术栈

| 层 | 技术 |
|-------|-----------|
| LLM | DashScope Qwen（turbo / plus / max / 235b） |
| Embedding | text-embedding-v3（1024 维） |
| Rerank | qwen3-rerank（回退到 RRF 分数） |
| 后端 | FastAPI + 4 个 async 路由 + python-multipart（文件上传） |
| 前端 | 原生 JS ES Modules（9 个模块）+ 6 层 CSS + 可折叠卡片 |
| 数据 API | Tushare、10jqka、Sina、EastMoney |
| 检索 | BM25 + ChromaDB（ANN）+ RRF + TextChunker + QueryParser（同义词 + 概念词典） |
| 检索评估 | 离线 IR 指标 harness（golden qrels + recall/MRR/NDCG/MAP + bm25/vector/fusion 分通道 + 基线回归锁） |
| 查询规划 | QueryPlanner（LLM 拆解，带 source/mode 的子查询） |
| 文档解析 | PyMuPDF（PDF，本地）+ qwen-vl-plus（图片多模态） |
| 图谱 RAG | LightRAG（PDF/图片 → 实体关系抽取 → 通过 function calling 查询图谱） |
| 领域词典 | DictionaryRegistry（JSON 可扩展：`data/dictionaries/*.json`） |
| 向量库 | ChromaDB（HNSW ANN，余弦距离，持久化存储） |
| 测试 | pytest —— 护栏驱动、按流水线步骤（273 通过，离线）；加固期 CI 门禁临时改为手动触发 |
| 可观测性 | Arize Phoenix（OpenInference/OTel，独立 `.venv-obs` 起服务 + 主应用 OTLP 外推，默认关闭；埋点收口 `financial_rag/observability/`） |

---

## 可观测性（Phoenix）

第一期已落地：**核心 RAG 链路**（LLM 调用 / 检索 / embedding / rerank）以 OpenInference span 推送到 Arize Phoenix，可视化 prompt·response·token·latency 与检索命中。

- **默认关闭**：`PHOENIX_ENABLED=false` 时埋点走 no-op，零副作用、不联网，不影响 273 个离线测试与 MOCK_MODE。
- **横切收口**：埋点集中在 `financial_rag/observability/tracing.py`（`observe_span` 上下文管理器），业务代码仅用 `with observe_span(...)` 包裹，未启用时透传 `yield None`。
- **环境隔离**：Phoenix 服务与主应用彻底分离——`arize-phoenix` 与主运行时存在硬性版本冲突，装进独立 `.venv-obs`（绝不进主环境）；主应用只需 `opentelemetry-sdk`（core）+ `opentelemetry-exporter-otlp-proto-http`（`observability` extra），经 OTLP/HTTP 把 span 推到外部 Phoenix 的 `/v1/traces`。
- **懒加载**：未装 `observability` extra（缺 OTLP 导出器）时，`init_phoenix` 优雅降级返回 False、不报错，主流程照常。
- **版本要求**：Python 3.11 需 `arize-phoenix>=21.0.0`——19.x/20.x 的 `trace/dsl/filter.py` 用 `MappingProxyType` 作 dataclass 默认值，py3.11 判定 unhashable、import 即崩（21.0.0 改用 `field(default_factory=...)` 修复）；已在 `requirements-observability.txt` 钉 `>=21.0.0`。

启用（本地可视化）：

```bash
# 1) 独立环境起 Phoenix 服务（首次建 .venv-obs，详见 requirements-observability.txt 注释）
scripts/start_phoenix.ps1            # Windows  |  scripts/start_phoenix.sh (Linux/macOS)
# 2) 主环境装 OTLP 导出器
uv sync --extra observability
# 3) 打开埋点开关并跑链路
export PHOENIX_ENABLED=true          # Windows PowerShell: $env:PHOENIX_ENABLED="true"
python -m financial_rag.main pipeline "商汤2024营收" -v
# 打开 http://localhost:6006 查看 LLM / RETRIEVER / EMBEDDING / RERANKER span
```

第二期（Agent 链 / Pipeline 5 阶段 / 工具调用 / 护栏）尚未埋点。

---

## 快速开始

```bash
# 1. 克隆并配置
git clone <your-fork-url>
cd financial-rag

# 2. 安装依赖（推荐 uv：自动建 .venv、按 uv.lock 精确复现）
uv sync --extra dev

# 激活 .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3.（无 uv 时的兼容旧流程）
# python -m venv myenv && 激活 && pip install -r requirements.txt

# 4. 配置 API key
cp .env.example .env
# 编辑 .env 并填入 DASHSCOPE_API_KEY

# 5. 启动 Web UI
python -m financial_rag.main web
# 打开 http://127.0.0.1:8000 —— 暗色主题，侧边栏导航
# 在"数据导入"面板拖拽上传 PDF/图片

# 6. 或使用 CLI
python -m financial_rag.main pipeline "这周 AI 圈发生了什么？"
python -m financial_rag.main pipeline "看下 600519 的 MACD" -v
python -m financial_rag.main toolcall -l   # 列出所有已注册工具
```

---

## 依赖管理（uv）

本项目用 [uv](https://docs.astral.sh/uv/) 做依赖与环境管理，`pyproject.toml` 是**单一真相源**，`uv.lock` 锁定全量版本保证可复现。四层治理：

- **拆分**：移除全项目 0 引用的 `llama-index` 冗余（自研 RAG 栈不依赖它），它曾拖进 `openai<3` 生态并与 Phoenix 冲突。
- **分组**：`core`（运行时 `dependencies`）+ `dev`（pytest/pytest-cov/pre-commit）+ `observability`（OTLP 导出器）三个 extras，按需安装。
- **隔离**：重依赖 `arize-phoenix` 与主运行时硬性冲突，装进独立 `.venv-obs`，绝不进主环境（见"可观测性"小节）。
- **可复现**：`uv.lock` 入库；`requirements.txt` 是 `uv export` 生成的锁定产物，供非 uv / 旧 pip 流程兼容，勿手改。

```bash
uv sync --extra dev              # 建 .venv 并装 core+dev（按 uv.lock 精确复现）
uv sync --extra observability    # 追加 OTLP 导出器（启用 Phoenix 埋点时需要）
uv run pytest tests/             # 在项目环境里跑测试
uv lock                          # 改了 pyproject 依赖后重新锁定
```

> 国内网络：uv **不读** pip 的镜像配置，需另配 `%APPDATA%\uv\uv.toml`（Linux/macOS：`~/.config/uv/uv.toml`）把默认索引设为清华镜像，否则直连 pypi.org 解析大依赖树会长时间卡住。

---

## 测试

```bash
python -m pytest tests/ -v    # 259 通过 —— 离线，无需 API key

# L1 检索评估 —— 分通道 IR 指标 + 基线回归锁（离线，无需 API key）
python -m financial_rag.eval.runner                    # 对比 baseline，融合核心指标回归即非零退出
python -m financial_rag.eval.runner --update-baseline  # 用本次结果刷新基线
```

这套测试**就是**上面描述的护栏：按步骤、从契约推导、证明效果、带回归锁。它**一次一个流水线步骤**地生长——完整 RAG 链覆盖是刻意保持未完成的（验证一步，再进下一步）。加固期内 CI 门禁临时改为手动触发（`workflow_dispatch`）。

测试只 mock **数据源**（Tushare、新闻 API）。检索被强制离线（BM25 + Jaccard）以保证确定性；LLM/embedding/rerank 仅在测试显式调用时才保持真实。

---

## 文档

- **[架构文档（中文）](docs/ARCHITECTURE.md)** —— 唯一权威基准：业务通路、Agent/工具清单、backlog

---

## 界面（V1.0）

<p align="center">
  <img src="docs/screenshots/agent_architecture.png" alt="Agent Architecture" width="100%">
  <em>3-Agent 链：Ingestion → Analysis → Scoring，由 AgentRouter 做基于意图的路由</em>
</p>

<p align="center">
  <img src="docs/screenshots/analysis.png" alt="Smart Query" width="100%">
  <em>查询面板：混合检索 + K 线技术分析</em>
</p>

<p align="center">
  <img src="docs/screenshots/kb_management.png" alt="Data Management" width="100%">
  <em>知识库管理：导入文件、构建索引（BM25 + embedding + RRF）、搜索/删除</em>
</p>

---

## 许可证

MIT
