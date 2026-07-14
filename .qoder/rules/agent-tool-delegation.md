---
description: Agent tool delegation rule — agents MUST NOT contain business logic, all heavy work goes through tool calls
globs:
  - "financial_rag/agents/**/*.py"
alwaysApply: true
---

# Agent Tool Delegation Rule

## Core Rule: Agents Are Lightweight Orchestrators, NOT Heavy Workers

An agent's `process()` method should contain **ONLY**:
1. Parameter extraction from `AgentContext` (routing/metadata)
2. `self.call_tool()` calls to delegate work
3. Result assembly into `AgentResult`

Everything else — data fetching, LLM calls, computation, file I/O, scoring — **MUST** live in a registered tool.

## What Counts as "Business Logic" (FORBIDDEN in Agents)

- ❌ Direct `tushare_client` / API imports and calls
- ❌ Direct `llm.chat()` / `llm.chat_with_tools()` calls
- ❌ Direct `HallucinationGuard` instantiation and calls
- ❌ Direct `PipelineScoreCard` instantiation and calls
- ❌ DataFrame manipulation, statistical computation
- ❌ File read/write (except IngestionAgent's text loading)
- ❌ Regex-based data extraction (beyond simple routing helpers like stock code parsing)

## What IS Allowed in Agents

- ✅ `self.call_tool("tool_name", **kwargs)` — delegating to tools
- ✅ `context.metadata.get(...)` — reading routing metadata
- ✅ `self._extract_stock_code(query)` — simple routing helpers (regex/keyword matching)
- ✅ `self.can_handle(context)` — intent checking
- ✅ Building `AgentResult` with `context_updates`
- ✅ `_build_sources()` / `_render_markdown()` — pure presentation formatting

## How to Add New Agent Capability

```python
# WRONG: Business logic in agent
class BadAgent(BaseAgent):
    def process(self, context):
        from financial_rag.tushare_client import fetch_kline  # ❌ direct import
        df = fetch_kline("600519.SH")                         # ❌ direct API call
        analysis = self._llm.chat(messages=...)               # ❌ direct LLM call
        return AgentResult(success=True, data=analysis)

# CORRECT: Delegate to tool
class GoodAgent(BaseAgent):
    def process(self, context):
        kline_data = self.call_tool("analyze_kline", ts_code="600519.SH")  # ✅ tool call
        analysis = self.call_tool("generate_report", data=kline_data)      # ✅ tool call
        return AgentResult(success=True, data=analysis)
```

## Tool Registration Pattern

When creating a new tool for an agent:
1. Create tool function in `tools/<module>_tools.py`
2. Define `FunctionDef` with name, description, parameters schema
3. Export as `*_TOOLS` list
4. Register in `tools/core.py` `create_financial_registry()`
5. Add LLM injection function if the tool uses LLM
6. Agent calls via `self.call_tool("tool_name", ...)`

## Agent `process()` Size Guideline

If an agent's `process()` method exceeds ~80 lines (excluding `_extract_*` routing helpers), it likely contains business logic that should be in a tool.
