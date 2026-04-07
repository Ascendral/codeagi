# LEARNING_RULES.md — What Counts as Real Learning

## The Core Question

"Did THE SYSTEM get smarter, or did I (the developer/AI assistant) get smarter?"

If the answer is "I did," that's not learning. That's coding.

## What Real Learning Looks Like

### Level 1: Memorization
- System stores a solution and retrieves it for an identical input.
- **Test:** Remove the solution from storage. Does accuracy drop? If yes, the system was using it.
- **Current state:** Not implemented. The `_try_*` functions are compiled in, not retrieved from memory.

### Level 2: Generalization
- System applies a learned pattern to a novel input it hasn't seen before.
- **Test:** Train on tasks A, B, C. Test on task D that shares the same underlying rule. Does it solve D without D-specific code?
- **Current state:** Not implemented. Each `_try_*` function is task-type-specific.

### Level 3: Strategy Discovery
- System discovers a new strategy without it being hand-coded.
- **Test:** Show the system a task type it has no `_try_*` function for. Does it synthesize a solution?
- **Current state:** `arc_learner.py` has 64 `_try_*` references. Needs audit to see if any of this is real discovery vs hand-coded.

### Level 4: Transfer
- System applies knowledge from domain A to improve in domain B.
- **Test:** Train on ARC color tasks. Does performance improve on ARC shape tasks?
- **Current state:** `brain_core/transfer.rs` exists but is not wired in. `connective/transfer_cortex.py` is untracked.

### Level 5: Self-Improvement
- System modifies its own strategy selection or weights based on outcomes.
- **Test:** Run eval. Let system observe failures. Run eval again. Did accuracy change without code changes?
- **Current state:** Not implemented.

## What Counts as Theater

| Claim | Reality | Verdict |
|-------|---------|---------|
| "System learned to solve X" | Developer wrote `_try_X()` | Theater |
| "Strategy ordering is learned" | Ordering is hardcoded or sorted by hit count from a static file | Depends — if hit counts update at runtime from real eval runs, it's Level 1 |
| "Memory affects behavior" | Memory files exist but nothing reads them at solve time | Theater |
| "Transfer cortex is working" | Code exists but isn't imported by solver | Theater |
| "Brain core handles retrieval" | Rust code compiles but Python never calls it | Theater |

## The Verification Checklist

Before claiming learning works:

1. **Identify the mechanism.** What code path does the learning?
2. **Show the data flow.** Input -> learning step -> stored knowledge -> retrieval -> changed behavior.
3. **Measure before.** Run eval without the learning. Record score.
4. **Measure after.** Run eval with the learning. Record score.
5. **Diff the results.** Which specific tasks changed? Why?
6. **Ablate.** Remove the learned knowledge. Does the score revert?

No measurement = no claim.

## Current Honest Assessment

The 341/1000 score is real. But it comes from 486 hand-coded `_try_*` functions.
That's a strong hand-coded baseline, not a learning system.

The gap between "hand-coded baseline" and "system that learns" is THE problem to solve.
Closing that gap is the only work that matters.
