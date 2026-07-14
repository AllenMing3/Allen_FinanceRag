---
description: Minimal change discipline — prevent scope creep and debug spirals on every task
globs:
  - "**/*.py"
alwaysApply: true
---

# Minimal Change Principle

## Core Rule: Touch ONLY What the Request Demands

Before writing any code, answer: **"What is the minimum set of files that MUST change to fulfill this request?"**

If a file isn't in that set, DO NOT modify it. Period.

## Pre-Change Checklist (Do This BEFORE Editing)

1. **Scope**: List exactly which files need changes and why. If >3 files, STOP and re-evaluate — the scope is probably too wide.
2. **Read first**: Read every target file fully before editing. Understand the existing pattern before changing it.
3. **No speculative fixes**: If you notice a "related issue" while working, NOTE it but do NOT fix it now. Separate task.
4. **No cascade renames**: If renaming a function/class would require updating 5+ files, keep the old name. Add the new one as an alias if needed.

## Anti-Patterns (DO NOT)

- ❌ "While I'm here, let me also update..." — No. Stay focused.
- ❌ "This docstring/comment is outdated, let me fix it" — Only if the request explicitly asks for doc updates.
- ❌ "Let me rename this to be more consistent" — Renames cascade. Avoid unless explicitly requested.
- ❌ "I should update all the tests/skills/docs too" — Only update docs if the request asks for it.
- ❌ "Let me refactor this to be cleaner" — Refactoring is a separate task. Never bundle it.
- ❌ Touching `__init__.py` exports for internal renames — Use aliases (`old_name = new_name`) in the source file instead.

## When You Catch Yourself Over-Changing

**STOP. Revert. Scope down.**

Ask: "If I only changed ONE file, would the request still be fulfilled?"
If yes → do only that one file.

## Error Handling During Implementation

If a change causes errors in 2+ files:
1. First: Is the approach wrong? Consider a simpler path.
2. Second: Can I contain the fix to the original file?
3. Only as last resort: Touch additional files, one at a time.

NEVER debug in a spiral of "fix one → break another → fix that → break a third".

## Documentation Updates

Only update README/USER_GUIDE/skills when:
- The user explicitly asks for it
- The change makes existing docs **factually wrong** (not just "could be better")

If docs just need "tweaking" — note it, don't do it.
