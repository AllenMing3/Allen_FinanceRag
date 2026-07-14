# 深度分析接入 Agent 链

## 目标

深度分析当前绕过 Agent 链（router 直接调 service），改为走 `AnalysisAgent -> ScoringAgent` 完整链路。和 RAG 查询对齐架构。

## 当前 vs 目标数据流

```
当前: /api/analyze/news  -> analyze_news_text() -> _run_hallucination_check(mode="analysis")
      /api/analyze/topic -> analyze_topic_research() -> _run_hallucination_check(mode="analysis")

目标: /api/analyze/news  -> AgentContext(intent="news")
        -> orchestrator.execute([AnalysisAgent, ScoringAgent])
            -> AnalysisAgent._run_deep_news_chain() -> call_tool("analyze_news_deep")
            -> ScoringAgent -> evaluate_pipeline_quality + check_hallucination(mode="analysis")
        -> 提取结果 -> 变换为前端 JSON

      /api/analyze/topic -> AgentContext(intent="deep_topic")
        -> orchestrator.execute([AnalysisAgent, ScoringAgent])
            -> AnalysisAgent._run_deep_topic_chain() -> call_tool("analyze_topic_deep")
            -> ScoringAgent -> 同上
        -> 提取结果 -> 变换为前端 JSON
```

---

## Task 1: AnalysisAgent 添加 deep_topic 意图 + 为 ScoringAgent 准备上下文

**文件**: `financial_rag/agents/analysis_agent.py`

### 1a: 添加 deep_topic 意图匹配

当前 `process()` 只匹配 `intent == "news" and context.parsed_data`，话题调研没有触发入口。

```python
# process() 中新增:
elif intent == "deep_topic":
    return self._run_deep_topic_chain(context)
```

### 1b: 深度分析链为 ScoringAgent 预加工上下文

`_run_deep_news_chain` 和 `_run_deep_topic_chain` 的 `context_updates["metadata"]` 中补充 ScoringAgent 需要的通用字段:

```python
"metadata": {
    # ScoringAgent 通用接口字段:
    "scoring_source_items": [{"text": news_text}] + kb_source_texts,  # 防幻觉 grounding
    "scoring_mode": "analysis",  # Guard mode: "rag" 或 "analysis"
    "scoring_text": full_analysis_text,  # 重建后的完整分析文本（含关键信号/风险等）
    # 业务字段（router 提取用）:
    "analysis_mode": "news",  # 或 "topic"
}
```

关键: 文本重建（`_build_full_analysis_text`）在 AnalysisAgent 里做，不在 ScoringAgent 里做。
AnalysisAgent 已经拿到了 `structured` 和 `analysis_text`，在这里重建完整文本最自然。

---

## Task 2: check_hallucination 工具支持 mode 参数

**文件**: `financial_rag/tools/scoring_tools.py`

当前 `check_hallucination()` 调用 `guard.check(output_text, sources)` 无 mode 参数，默认走 rag 模式。深度分析输出格式是 `【关键信号】` 不是 `[N]` 引用，需要 `mode="analysis"`。

改动:
- 函数签名: `check_hallucination(output_text, source_items=None, mode="rag")`
- 调用: `guard.check(output_text, sources, mode=mode)`
- FunctionDef parameters 新增 `"mode"` 字段

---

## Task 3: ScoringAgent 重构为通用公共评分 Agent

**文件**: `financial_rag/agents/scoring_agent.py`

### 设计原则

- ScoringAgent 是**公共能力**，任何 feature 都能接入
- Agent 只做 3 个 tool call，不写实现逻辑
- 所有预加工由上游（AnalysisAgent 或 router）通过 `context.metadata` 传入

### 通用接口契约

ScoringAgent 从 `context.metadata` 读取以下**通用字段**:

| 字段 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `scoring_source_items` | `list[dict]` | 防幻觉 grounding 源 | AnalysisAgent 或 router |
| `scoring_mode` | `str` | Guard mode: `"rag"` / `"analysis"` | 上游按场景设置 |
| `scoring_text` | `str` | 待校验的完整文本 | AnalysisAgent 重建 或 final_answer |
| `retrieved_items` | `list[dict]` | 检索结果（RAG 场景） | Pipeline phase 2 |
| `fetched_data` | `list[dict]` | 获取数据（RAG 场景） | Pipeline phase 1 |
| `fill_stats` | `dict` | 槽位填充统计（RAG 场景） | Pipeline phase 4 |

### process() 伪代码（保持轻量）

```python
def process(self, context):
    metadata = context.metadata

    # 1. 读取通用字段（有就用，没有就跳过）
    source_items = metadata.get("scoring_source_items")  # 优先用显式传入
    if not source_items:
        # RAG 场景: 从 retrieved_items 构建
        source_items = [{"text": it["text"]} for it in metadata.get("retrieved_items", []) if it.get("text")]

    guard_mode = metadata.get("scoring_mode", "rag")
    check_text = metadata.get("scoring_text") or context.final_answer or ""

    # 2. 三个 tool call（不变）
    pipeline_scores = self.call_tool("evaluate_pipeline_quality", ...)
    hallucination_check = self.call_tool("check_hallucination",
        output_text=check_text, source_items=source_items, mode=guard_mode)
    report = self.call_tool("generate_score_report", ...)

    # 3. 组装 AgentResult
    return AgentResult(...)
```

核心改动: 新增 `scoring_source_items` / `scoring_mode` / `scoring_text` 三个通用字段的读取逻辑，优先于 RAG 场景的 `retrieved_items`。这样任何 feature 只要往 metadata 塞这三个字段就能接入评分。

---

## Task 4: analysis_router 改用 Agent 链

**文件**: `financial_rag/api/analysis_router.py`

这是最大的改动。当前 `/api/analyze/news` 和 `/api/analyze/topic` 直接调 service + router 级 HallucinationGuard。改为通过 orchestrator 执行 Agent 链。

### 4a: 新增辅助函数 `_run_analysis_chain()`
allen//别老加函数，加工具！

```python
def _run_analysis_chain(intent: str, raw_input: str, metadata: dict) -> dict:
    """通过 Agent 链执行深度分析，返回结构化结果"""
    orch = _state["orchestrator"]
    orch.set_pipeline(["AnalysisAgent", "ScoringAgent"])

    context = AgentContext(
        raw_input=raw_input,
        parsed_data=[{"text": raw_input}] if intent == "news" else None,
        metadata={"intent": intent, **metadata},
    )

    exec_result = orch.execute(raw_input, context=context)
    return _extract_chain_result(exec_result, intent)
```

### 4b: 新增结果映射函数 `_extract_chain_result()`

从 `ExecutionResult` 中提取 AnalysisAgent 和 ScoringAgent 的输出，映射为前端期望的 JSON 格式:
```python
# 前端期望:
{
    "assessment": "bullish/bearish/neutral",
    "analysis": "分析文本",
    "structured": {...},
    "confidence": "high/medium/low",
    "hallucination": {overall_score, risk, passed, layers},
    "metrics": {...},
    "entities": {...},
    "doc_type": "...",
    "kb_sources": [...],
    "session_id": "...",
}
```

### 4c: 改写 `/api/analyze/news`

```python
result = _run_analysis_chain(
    intent="news",
    raw_input=text,
    metadata={"query": req.query, "analysis_mode": "analysis"},
)
# 会话创建、KB 保存等逻辑保持 router 内联
```

### 4d: 改写 `/api/analyze/topic`

```python
result = _run_analysis_chain(
    intent="deep_topic",
    raw_input=topic,
    metadata={"topic": topic, "max_news": req.max_news, "analysis_mode": "analysis"},
)
# 会话创建、KB 保存、metadata 存储等逻辑保持 router 内联
```

### 4e: 清理 router 级 HallucinationGuard

移除 `_run_hallucination_check()` 和 `_build_full_analysis_text()` 的直接调用（已由 ScoringAgent 处理）。保留这两个函数作为 fallback（agent chain 失败时降级使用）。

---

## Task 5: 验证

- 全量测试通过
- 手动验证: 新闻解读 -> 前端显示 verdict + 评分面板 + 各层分数
- 手动验证: 话题调研 -> 前端显示子话题 + 评分面板

---

## 涉及文件

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `agents/analysis_agent.py` | 修改 | 添加 deep_topic 意图 + context_updates 补充 |
| `tools/scoring_tools.py` | 修改 | check_hallucination 加 mode 参数 |
| `agents/scoring_agent.py` | 修改 | 适配深度分析上下文 + mode 传递 |
| `api/analysis_router.py` | 修改 | 核心改动: agent chain 替代直接调用 |

## 不改的文件

- `services/analysis.py` -- 核心分析逻辑不变，被 analyze_news_deep/analyze_topic_deep 工具调用
- `tools/analysis_tools.py` -- 工具封装不变
- `static/modules/analyze.js` -- 前端不变，接收相同 JSON 格式
- `core/pipeline.py` -- Pipeline 不变，深度分析不走 Pipeline
- `core/orchestrator.py` -- 编排器不变，被 router 直接使用
