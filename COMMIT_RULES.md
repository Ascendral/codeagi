# COMMIT_RULES.md — Git Discipline

## Commit Message Format

Every commit message MUST start with a category tag in brackets:

```
[CATEGORY] Short description of what changed
```

## Categories

| Tag | Meaning | Example |
|-----|---------|---------|
| `[SOLVE]` | New strategy that solves additional ARC tasks | `[SOLVE] Add flood_fill_bounded: 342/1000 (+1)` |
| `[FIX]` | Bug fix that doesn't change score | `[FIX] Correct off-by-one in grid crop` |
| `[REFACTOR]` | Code restructuring, no behavior change | `[REFACTOR] Extract color counting to shared util` |
| `[EVAL]` | Changes to evaluation/benchmark code | `[EVAL] Add per-category accuracy breakdown` |
| `[LEARN]` | Changes to learning infrastructure | `[LEARN] Wire strategy hit counts into runtime state` |
| `[INFRA]` | Build, deploy, tooling, CI | `[INFRA] Add pre-commit hook for _try_ detection` |
| `[DOCS]` | Documentation changes | `[DOCS] Update ARCHITECTURE.md with live code audit` |
| `[TEST]` | Test additions or changes | `[TEST] Add regression tests for gravity strategies` |
| `[SECURITY]` | Security fixes | `[SECURITY] Remove hardcoded credentials from push.sh` |

## Rules

### 1. Score Changes Must Include Numbers
If a commit changes the ARC score, the message MUST include:
- New score: `342/1000`
- Delta: `(+1)` or `(-3)`

Bad: `[SOLVE] Add new strategy`
Good: `[SOLVE] Add flood_fill_bounded: 342/1000 (+1)`

### 2. No Unverified [SOLVE] Commits
Before committing with `[SOLVE]`:
- Run `python3 -m klomboagi.evals.arc_eval` (or equivalent).
- Include the actual score in the commit message.
- If you didn't run the eval, use `[WIP]` instead.

### 3. No _try_ Functions in [LEARN] Commits
A `[LEARN]` commit must NOT introduce new `_try_*` functions.
That's a `[SOLVE]` commit wearing a costume.

The pre-commit hook enforces this.

### 4. One Concern Per Commit
Don't mix categories. If you're adding a strategy AND fixing a bug, make two commits.

### 5. No "WIP" on Main
`[WIP]` commits should only exist on feature branches.
Squash or rewrite before merging to main.

## Git Hooks

### pre-commit
- Blocks new `_try_*` function definitions in `[LEARN]` commits.
- Warns about `_try_*` in any commit (informational).
- Checks for hardcoded credentials patterns.

### commit-msg
- Rejects commits without a valid category tag.
- Validates that `[SOLVE]` commits include a score.

## Branch Strategy

- `main` — working code, all tests pass, score verified.
- `feature/*` — experimental work, WIP allowed.
- `learn/*` — learning system development, no new `_try_*` allowed.
