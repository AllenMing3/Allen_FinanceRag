---
name: dev-workflow
description: 开发工作流规则 — Plan 用中文、用户确认后再写代码、新功能前先测老功能、最后验证、接受不完美、API 限速、最小改动。规划任务、实现功能、调试、决定验证时机时使用。
---

# 开发工作流规则

## 血泪教训（必须遵守）

### 1. Plan 必须用中文

所有计划、方案、任务描述一律用中文编写。代码注释和变量名可以用英文，但**给人看的方案必须中文**。

### 2. 用户确认后再写代码

**Plan → 用户 Review → 确认 → 才开始写代码**。绝不跳过确认直接开干。

```
❌ 错误流程：收到需求 → 直接写 200 行代码 → 发现方向错了 → 全部推翻
✅ 正确流程：收到需求 → 写中文 Plan（含文件清单和方案） → 用户确认/修改 → 按 Plan 写代码
```

**例外**：用户明确说"赶紧做"、"直接加"、不需要确认的简单改动。

### 3. 新功能开发前先测老功能

大幅开发新功能之前，**先跑一遍现有测试**，确认老功能没坏。

```
❌ 错误流程：加新功能 → 跑测试 → 发现老功能也挂了 → 分不清是新代码还是老代码的问题
✅ 正确流程：跑现有测试（全绿）→ 加新功能 → 再跑测试 → 如果有红，一定是新代码的问题
```

**什么时候必须做**：
- 改动涉及核心模块（Agent、Pipeline、Guard、Retriever）
- 改动涉及共享基础设施（config、tools/core、services）
- 新增 >3 个文件的大功能

## 核心原则

### 4. 最后验证，不在开发中途验证

全部实现完成后，**一次性**跑验证。不要在开发中间反复验证。

- 不要每改一个文件就跑测试
- 不要后端开发到一半就开浏览器检查 UI
- 不要反复执行同一个命令"确认"结果
- 信任第一次输出，只有明显有问题才重新验证

### 5. 前端验证最后做

顺序是：**后端逻辑 → API 端点 → 前端 UI → 浏览器验证**

- 先完成所有后端工作（Agent、Tool、Pipeline、API）
- 最后才验证前端
- 后端没完成时不要切到浏览器测试

### 6. 接受不完美

不是每个差异都需要立即修复：

- 非关键差异可以先记下，后续处理
- 初始实现时不要打磨边缘情况
- 记在注释或任务笔记里，下次再修

### 7. API/网络依赖处理

依赖外部 API 或不稳定网络时：

- 加限速（尊重 API 配额）
- 优雅失败 — 记日志继续走，不要崩溃或死循环重试
- 标记完成状态 — 哪些成功哪些失败，继续走

### 8. 最小改动优先

修改现有代码时：

- 只改当前任务需要的部分
- 不要顺手重构无关代码
- 不要加"顺便优化"的改动

## 任务执行模式

```
1. Plan（中文）— 列出要改的文件，确认方案
2. 用户确认 — 等用户 review 并确认/修改
3. 跑现有测试（如果是大改动）— 确认基线是绿的
4. 实现 — 按计划改代码，不做中间验证
5. 验证一次 — 最后跑测试/检查，一遍过
6. 汇报 — 总结做了什么，记下已知问题
```

## 反模式（禁止）

| 禁止 | 替代做法 |
|------|----------|
| 每改一个文件就跑测试 | 全部改完，最后跑一次 |
| 每改一个 API 就开浏览器 | 后端全部完成后再检查前端 |
| API 调用失败重试 5 次 | 记日志、加限速、继续走 |
| 顺手重构"乱"代码 | 记下来，单独任务做 |
| 重复执行同一命令"再确认" | 信任第一次结果 |
| 收到需求直接写 200 行代码 | 先写中文 Plan，等确认 |
| 加新功能不跑老测试 | 先跑老测试确认基线 |

## 范围控制

实现功能时：
1. 列出需要改的文件
2. 改那些文件
3. 停下来 — 不扩大范围，除非用户确认
---
name: dev-workflow
description: Development workflow rules — verify last (not during development), accept imperfection, frontend verification last, API rate limiting, prefer minimal changes. Use when planning tasks, implementing features, debugging, or deciding verification timing.
---

# Development Workflow Rules

These rules govern how development tasks should be approached and executed in this project.

## Core Principles

### 1. Verify Last, Not During Development

**After** all implementation is complete, run verification **once**. Never verify mid-implementation.

- Do NOT run tests after every small change
- Do NOT open the browser to check UI during backend development
- Do NOT repeatedly execute the same command to "confirm" results
- Trust the first output; only re-verify if something is clearly broken

### 2. Frontend Verification Comes Last

The order is: **Backend logic → API endpoints → Frontend UI → Browser verification**

- Complete all backend work (agents, tools, pipeline, API) first
- Only then verify the frontend
- Never switch to browser testing while backend work is in progress

### 3. Accept Imperfection

Not every discrepancy needs immediate resolution during development:

- Non-critical differences can be noted and addressed later
- Don't stop to polish edge cases during initial implementation
- Record issues in comments or task notes, fix in a follow-up pass

### 4. API/Network Dependency Handling

When code depends on external APIs or unstable networks:

- **Add rate limiting** before making calls (respect API quotas)
- **Accept failures gracefully** — log and continue, don't crash or retry-loop
- **Mark completion status** — indicate what worked and what didn't, move on

### 5. Prefer Minimal Changes

When modifying existing code:

- Change only what's necessary for the current task
- Don't refactor unrelated code unless explicitly requested
- Don't add "while I'm here" improvements

## Task Execution Pattern

```
1. Plan — identify files to change, confirm approach with user
2. Implement — make all changes without intermediate verification
3. Verify once — run tests/checks at the end, one pass
4. Report — summarize what was done, note any known issues
```

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|-----------|
| Run tests after every file edit | Implement all changes, then run tests once |
| Open browser after each API change | Finish all backend work, then check frontend |
| Retry failed API calls 5 times | Log failure, add rate limit, continue |
| Refactor "messy" code nearby | Note it, fix in a separate task |
| Re-run same command to "double check" | Trust the first result |

## Scope Control

When implementing a feature:
1. List exactly which files need changes
2. Make those changes
3. Stop — don't expand scope without user confirmation
