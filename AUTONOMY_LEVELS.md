# AUTONOMY_LEVELS.md — Honest Assessment Scale

## Purpose

A system for measuring how autonomous KlomboAGI actually is.
Not how autonomous we want it to be. How autonomous it IS, today, with evidence.

## Level 0: Manual
**Definition:** Human writes all logic. System executes what's written.

**Evidence required:** N/A — this is the default.

**Current state: THIS IS WHERE WE ARE.**
- Every `_try_*` function was written by a human or AI assistant.
- Strategy ordering is determined at code-write time.
- No runtime adaptation. No self-modification. No autonomous decisions.
- Score: 341/1000, entirely from hand-coded strategies.

## Level 1: Assisted
**Definition:** System has persistent state that influences execution order, but all strategies are still hand-coded.

**Evidence required:**
- Show runtime state that changes between runs.
- Show that changed state produces different execution order.
- Show that different order produces different (better) outcomes.

**Current state: NOT REACHED.**
- `runtime/state/` directory exists but is not proven to be read by the solver.
- Strategy ordering in `arc_smart_solver.py` appears static.

## Level 2: Adaptive
**Definition:** System modifies its own behavior based on observed outcomes without code changes.

**Evidence required:**
- Run eval A. System observes results. Run eval B. Behavior changed.
- No code was modified between A and B.
- Show the specific mechanism (weights updated, strategy re-ranked, etc).

**Current state: NOT REACHED.**

## Level 3: Self-Extending
**Definition:** System creates new strategies or modifies existing ones without human intervention.

**Evidence required:**
- System encounters novel task type.
- System synthesizes a solution approach not present in codebase.
- Solution works on held-out test cases.
- No human wrote the solution.

**Current state: NOT REACHED.**
- `brain_core/learning.rs` and `arc_learner.py` exist but are not proven to generate novel strategies at runtime.

## Level 4: Self-Correcting
**Definition:** System identifies its own failures, diagnoses root causes, and fixes them.

**Evidence required:**
- System fails task X. System analyzes why. System modifies approach. System retries. System succeeds.
- All of this happens without human intervention in a single session.

**Current state: NOT REACHED.**

## Level 5: Self-Improving
**Definition:** System's overall capability increases over time through its own operation, not through human development.

**Evidence required:**
- Eval score at time T1.
- System runs autonomously for period P (no human code changes).
- Eval score at time T2 > T1.
- Improvement is statistically significant and reproducible.

**Current state: NOT REACHED.**

## How to Use This Scale

1. **Never claim a level you can't prove.** "We have infrastructure for Level 2" means nothing. Either we're at Level 2 with evidence, or we're not.

2. **Measure the transition.** Moving from Level 0 to Level 1 requires:
   - Wiring runtime state into the solver's strategy selection.
   - Showing that state persists between runs.
   - Showing that persistence changes outcomes.

3. **One level at a time.** Don't build Level 3 infrastructure until Level 1 is proven.

## Current Status Summary

| Level | Status | Evidence |
|-------|--------|----------|
| 0 - Manual | CURRENT | 486 hand-coded _try_* functions |
| 1 - Assisted | NOT REACHED | Runtime state exists but not wired in |
| 2 - Adaptive | NOT REACHED | No runtime behavior modification |
| 3 - Self-Extending | NOT REACHED | No strategy synthesis |
| 4 - Self-Correcting | NOT REACHED | No autonomous failure recovery |
| 5 - Self-Improving | NOT REACHED | No autonomous capability growth |
