---
description: Verify approach before committing — prevent wrong-path debug spirals
globs:
  - "**/*.py"
alwaysApply: true
---

# Verify Before Commit

## The Problem

Going down the wrong implementation path → hitting errors → patching → more errors → 30 minutes wasted on what should have been a 5-minute task.

## Rule: Test the Approach in Isolation FIRST

Before writing any production code that calls external APIs or changes data flow:

1. **Test the API endpoint** in a one-liner or scratch script BEFORE building the full implementation
2. **Verify the response structure** matches your assumptions (field names, types, nesting)
3. **Only THEN** write the production code

### Example (Good)

```bash
# Step 1: Quick API probe (30 seconds)
python -c "import httpx; r=httpx.get('https://api.example.com/data'); print(r.status_code, r.text[:200])"

# Step 2: API works, response confirmed → now write the real code
```

### Example (Bad)

```python
# Writing 100 lines of code based on ASSUMED API behavior
# Then testing → 400 error → rewrite → 404 → rewrite again → ...
```

## When Changing Existing Code

Before modifying a function:
1. **Understand who calls it**: `grep` or search for all callers first
2. **Check if the return type/shape changes**: If yes, ALL callers need to handle it
3. **Prefer additive changes** over breaking changes: Add a new param with a default, don't remove/rename existing params

## Decision Points

| Situation | Action |
|-----------|--------|
| Not sure if API works | Test with one-liner first |
| Change affects return type | Search all callers before editing |
| Two approaches possible | Pick the one touching fewer files |
| Hitting second error on same change | STOP. Re-evaluate approach entirely. |
| Third file needs fixing | WRONG APPROACH. Revert and rethink. |

## The 3-Error Rule

If you hit **3 errors in a row** while implementing a single change:
1. **STOP coding**
2. **Re-read the requirements** — did you misunderstand?
3. **Pick a completely different approach** — the current path is wrong
4. **Ask the user** if the approach is correct before continuing
