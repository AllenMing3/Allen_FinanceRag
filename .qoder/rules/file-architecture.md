# File Architecture — Size Limits and Separation of Concerns

## The Problem

Writing everything into one file → 1000+ lines → can't maintain → split later → wasted effort. This is the #1 anti-pattern in this codebase.

## Rule: Plan File Placement BEFORE Writing Code

Before adding any feature, answer: **"Does this belong in an existing file, or does it need its own?"**

### File Size Limits

| Threshold | Action |
|-----------|--------|
| **500 lines** | Soft limit — stop and evaluate. Can this be split? |
| **800 lines** | Hard limit — MUST split before adding more. No exceptions. |
| **300 lines** | Target for service modules (pure logic, no HTTP) |

### Separation of Concerns (Enforced)

Every feature that touches HTTP **must** split into layers from day one:

```
web.py          → HTTP only: request validation, call service, return response (thin shell)
services/*.py   → Pure business logic: takes deps as kwargs, returns dicts, no FastAPI imports
persistence.py  → Storage I/O: JSON read/write, file operations
```

**Anti-patterns (DO NOT):**
- ❌ Writing business logic inside an endpoint function
- ❌ Adding a new endpoint with 50+ lines of inline logic to web.py
- ❌ Putting database/file operations in the same file as HTTP handlers
- ❌ "I'll refactor later" — if it's >50 lines of logic, it goes in services/ NOW

### New Feature Checklist

When adding a new capability (e.g., a new analysis type, a new data source):

1. **Service first**: Create or extend a file in `services/` with pure functions
2. **Endpoint second**: Add a thin wrapper in `web.py` (5-15 lines max per endpoint)
3. **Test the service**: Test the pure function, not the HTTP endpoint
4. **Check size**: If `web.py` exceeds 800 lines, extract more endpoints to routers or services

### Function Length

| Length | Action |
|--------|--------|
| < 30 lines | Normal |
| 30-60 lines | Acceptable for complex logic |
| > 60 lines | MUST extract helper functions |
| > 100 lines | Wrong. Split immediately. |

### When In Doubt

Ask: **"If someone reads this file for the first time, can they understand it in 2 minutes?"**

If the answer is no → the file is too big or does too many things.

## Decision Points

| Situation | Action |
|-----------|--------|
| New endpoint with >20 lines of logic | Put logic in `services/`, endpoint stays thin |
| File approaching 500 lines | Stop. Split now, not later. |
| Feature needs 3+ helper functions | New file in `services/` or `tools/` |
| Mixing HTTP + business logic + storage | Wrong. Three files minimum. |
| "Quick prototype" in one file | Fine for <100 lines. Over that, plan structure first. |
