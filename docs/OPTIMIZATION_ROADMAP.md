# FinRAG 优化方向路线图（初稿 · 待交叉 review）

> 状态：**方向层结论**，代码未动。本文汇总本轮四个重点（调度层 / 数据存储 / 可靠性 agent /
> 原始资料拆分+prompt）的审查发现与初步优化方向，供交叉 review。
> 详细逐条 Backlog 见 [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) §8。
> 证据标注：`（探针实测）` = 已用临时脚本跑真实代码路径验证；`file:line` = 静态代码定位。

---

## 0. 贯穿全局的判断

四个重点盘下来，暴露的不是零散 bug，而是**三个"本该唯一却散落多处"的收敛缺口**：

| 缺口 | 现状 | 后果 |
|------|------|------|
| 可靠性评分 | `HallucinationGuard.check()` 有 **4 个**各自为政的调用点 | 同一输出走不同入口分数不可复现、通路 B 双重评分 |
| 检索→喂 AI 的上下文组装 | **3 套**截断策略各截各的（150 / 800 / 不截断） | 同一 chunk 走不同通路信息量差 5 倍以上 |
| 数据存储 | **4 份真相**（kb_docs / news_meta / stats / version）各写各的 | 内存单例整表重写 + 无锁并发 → 丢数据 |

**主线方针**：业务通路第一优先级，先把"散"收敛成"唯一真相 / 唯一入口"，效能与历史债降级挂起。

---

## 方向一：数据存储 → 迁移 PostgreSQL（已拍板 · 待开工）

**病根**
- 启动时整表 load 进内存 list 单例，每个请求 mutate 后整表 dump 回文件，普通请求全程无锁
  → FastAPI `asyncio.to_thread` 并发下 last-writer-wins，一个请求拿旧副本整表覆盖抹掉另一个
  请求刚写的数据（`news_metadata.json` 曾莫名 -1361 行即此症状）。
- kb_docs / news_meta / stats / version **四份真相**靠计数校验勉强对齐，易漂移。

**方向**
- 核心先行：kb_docs / news_meta / stats / version 入 PG 做**唯一真相**；
  Chroma / LightRAG / bm25 降为**可重建派生索引**（坏了可从 PG 重建）。
- `persistence.py` 换实现、留签名 —— ~18 处调用点零改动。

**前置依赖**：PG 实例（本地/Docker/云）+ 驱动（psycopg 或 SQLAlchemy）+ Alembic 迁移 + `config.py`/`.env` 连接串。

**Backlog**：§8 P1-9。

---

## 方向二：可靠性层 → 抽 ReliabilityService 统一入口（已拍板 · 待开工）

**病根**
- `HallucinationGuard.check()` 现有 **4 个调用点**，每处自拼 sources、自定 mode、自决触发：
  1. 链内 `scoring_tools.check_hallucination`（走 ScoringAgent）
  2. `analysis_router._run_hallucination_check`（新闻/话题 fallback，硬编码 `mode=analysis`）
  3. `kb_router` /kb-query（默认 `mode=rag`）
  4. `pipeline._phase_evolve`
- 后果：同一输出走不同入口**分数不可复现**；通路 B 上 ①④ 对同一输出**双重评分**；
  改策略需同步 4 处，极易漂移（`serializer` 注释仍写"统一 3 个调用点"，实为 4）。
- **P0 附带**：可靠性层自己会被**静默跳过** ——
  ① `ScoringAgent.can_handle = bool(final_answer or intermediate_findings)`（scoring_agent.py L51-53），
  上游链忘写回 final_answer → orchestrator 无痕 continue，整层评分+防幻觉消失且无"未校验"提示；
  ② `check_hallucination` 中 `if not check_text: return {}`（scoring_agent.py L101-102）。
  二者均违反"防幻觉禁止静默跳过"规范（该规范只在 Guard 内部 skip 落实，未覆盖"整层该不该跑"）。

**方向**
- 抽 `ReliabilityService` 封装 `check() + mode 决策 + source 组装 + 序列化`，
  其余 3 处改为调它，ScoringAgent 亦复用。**六层 Guard 引擎不动，只收敛接线。**
- 堵静默跳过：can_handle 兜底 + 空文本走显式 `skipped` 标记而非 `return {}`。

**Backlog**：§8 P0-2、P1-5。

---

## 方向三：检索→组装 → 统一 context-builder（本轮新识别 · 收益最大）

> **关键**：§8 的 P0-3（grounding 腰斩）、P1-10（三套组装不一致）、P1-5（可靠性入口收敛）
> 其实是**同一个根** —— "检索结果 → 喂给 AI"这一步没有统一的 context-builder，三处各截各的。
> 建议未来开工时**合并成一个改造**，一次性根治。

**病根（探针实测）**
- 深度分析路径把 KB grounding 砍到 `kb_sources[:3]` 且每条 `s['text'][:150]`
  （analysis.py L388 / L512）：实测召回 **3286 字** chunk 只喂进 **450 字**，
  **保真率 13.7%**，且每条断在半句/半词（如"…提供智"、"…后续需关注三个"）。
- 连锁后果：AI 几乎拿不到真实 KB 上下文（只能靠新闻原文 `text[:3000]` + 先验）→ **幻觉温床**；
  而 HallucinationGuard 又用**同一批 150 字残片**当 grounding 打分 → answer 与"来源"同源于
  贫瘠输入 → **防幻觉分数系统性虚高、自我欺骗**。
- 三套截断：KB问答（slot_filler 不截断）/ 深度分析（每条 150）/ 新闻综合（每条 800，总 12000）。
- 三套 prompt：`prompts.py` 富模板（CoT+anti_hallucination+rubric+few-shot）/
  `analysis.py` 内联富 prompt（L390-448，与前者风格重复）/ `slot_filler` 极简 system（L96-99）。

**方向**
- 建**统一 context-builder**：按 **token 预算**（而非固定 150 字）组装 grounding，
  优先保整句/整段，三条通路共用；prompt 收敛到 `prompts.py` 单一来源。
- 配套 P0：embedding 缓存 hash 改**全文**或 `(source_id + chunk_id)`，根治 `text[:500]` 撞键
  导致的向量污染（探针实测：两 chunk 前 500 字相同即得同一 hash `ec398a…`，后者复用前者向量）。
- 配套 P1：chunk 配置统一到单一源（现散落 factory 1500/100/80、chunk_tools default 500/50、
  kb_router `req.chunk_size`）。

**Backlog**：§8 P0-3、P0-4、P1-10、P1-12。

---

## 方向四：调度层 → 通路 B 接线修复 + 入口收敛（已识别）

**病根**
- 通路 B（pipeline 深度研究）：`_phase_process` 只设 `result.agent_exec_result`，
  **从不设 `result.final_output`** → Phase4"Agent 已产出则跳过槽位填充"守卫（pipeline.py L453）
  成死代码 → SlotFiller 总是覆盖 Agent 分析；若 filler 为 None 则通路 B 返回空。
- 三条 AnalysisAgent 入口未收敛（链 / `_run_analysis_via_agent_chain` / api_kline 直连）。
- 编排器共享单例，`set_pipeline()` + `self.context` 为可变全局状态，并发请求互相覆写
  （属效能/竞态，按方针暂降级挂起）。

**方向**
- 补齐 Phase3→Phase4 的 `final_output` 传播，让槽位守卫真正生效。
- 三条 AnalysisAgent 入口收敛为一条。

**Backlog**：§8 P0-1、P1-7、P1-8。

---

## 明确降级 / 挂起（不阻塞业务）

- 假分数换真实质量信号（`evaluate_pipeline_quality` / `_phase_evolve` 硬编码 0.9/0.85/0.8；RAGAS 基线已建）。
- L5/L6 触发条件自我掩盖（LLM 纠错层仅在 `rule_score < 0.85` 触发，而 L1 阈值极低易假阳性顶高 rule_score）。
- 重试机制失效（BaseAgent.run 吞异常，max_retries 永不触发）。
- async 化、死代码清理（PARALLEL/CONDITIONAL、MessageBus 等）、orchestrator.history 无界增长。
- chunk 边界/位置追溯损坏（探针实测 chunk_start 非单调、overlap 后超 chunk_size）。
- 测试套件按 5 通路重写（迭代期暂停全量 pytest 门禁）。

---

## 建议开工优先级

| 序 | 方向 | 理由 |
|:--:|------|------|
| 1 | **方向三 context-builder** | 收益最大：一次性根治 grounding 腰斩 + 三套不一致 + 喂料碎渣，直接提升所有通路输出质量 |
| 2 | **方向二 ReliabilityService** | 与方向三共享"统一 source 组装"，可紧邻做 |
| 3 | **方向一 PostgreSQL** | 独立大件，需先备 PG 环境 |
| 4 | **方向四 调度层接线** | 通路 B 修复 |

---

## 附：本轮审查方法说明（供 review 校准）

- 结论均基于**阅读真实运行代码**（非文档/记忆臆断），关键处交叉验证。
- 标注"探针实测"的结论用临时脚本跑真实代码路径得到（脚本用完即删，未留工作区噪音）：
  - grounding 保真率 13.7%（monkeypatch `LLMCaller.call_json` 拦截真实 prompt）
  - embedding 缓存撞键（`EmbeddingCache._hash` 对前 500 字相同的两 chunk 返回同 hash）
  - chunk_start 非单调 / overlap 后超 chunk_size（`TextChunker.split_by_doctype` 实跑）
- 欢迎 review 时对上述任一结论提出反证或补充。
