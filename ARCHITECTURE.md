# ARCHITECTURE.md — System State of Truth

Last audited: 2026-04-06

## What KlomboAGI Actually Is (Right Now)

An ARC-AGI-1 solver built on hand-coded pattern-matching strategies.
Not yet a learning system. Not yet autonomous. That's the goal, not the current state.

## What's Live and Running

### `klomboagi/reasoning/arc_smart_solver.py`
- **The main solver.** 636 `_try_*` functions (pattern-specific solvers).
- Currently at ~341/1000 ARC-AGI-1 tasks (34.1%).
- This is the workhorse. Every score improvement comes from here.

### `klomboagi/reasoning/arc_solver.py`
- Base solver, 251 `_try_*` references. Used by smart_solver.

### `klomboagi/reasoning/` (all `arc_*.py` files)
- Feature extraction, object detection, grid ops, region analysis, tiling, gravity, composition.
- **These are live.** They're called by the solver pipeline.

### `klomboagi/evals/arc_eval.py`
- Runs the solver against ARC-AGI-1 dataset via arckit.
- Reports accuracy, per-task results, strategy used, timing.
- **This is the benchmark.** The only number that matters.

### `deploy/push.sh`
- Deploys to remote MacBook Air at 192.168.68.53.
- **WARNING: Contains hardcoded credentials.** Needs cleanup.

### `deploy/com.klomboagi.server.plist`
- launchd service definition for the runtime server.

## What Exists But Is Not Proven Live

### `klomboagi/connective/` (untracked)
- `nexus.py`, `transfer_cortex.py`, `investigation.py`, `runtime_bridge.py`
- Appears to be cross-module integration layer.
- **Status: untracked, not imported by solver pipeline. Needs verification.**

### `brain_core/` (Rust)
- `learning.rs`, `plan_search.rs`, `retrieval.rs`, `scoring.rs`, `transfer.rs`
- **Status: not wired into Python solver. Not called at runtime. Aspirational.**

### `runtime/` (untracked)
- `cache/`, `logs/`, `queue/`, `state/`, `temp/`, `working_memory/`
- Directory structure exists. **Not proven to be read/written by active code.**

### `long-term/` (untracked)
- `archives/`, `backups/`, `checkpoints/`, `datasets/`, `evals/`, `experiments/`, `integrity/`, `manifests/`, `memory/`, `world/`
- **Status: directory scaffolding. Not proven to be used by solver.**

### `workspace/`
- Empty.

### `evals/`, `scripts/`
- Contain only `__pycache__/`. No source files.

## Architecture from ARCHITECTURE.md (Original) vs Reality

The original ARCHITECTURE.md describes 8 layers:
Executive, Memory, World Model, Reasoning, Action, Learning, Safety, Evaluation.

**What actually exists in code:**
- **Reasoning layer:** YES. `klomboagi/reasoning/` is real and working.
- **Evaluation layer:** PARTIAL. `arc_eval.py` exists and works.
- **Everything else:** Described but not implemented as running code.

This is not a criticism. It's where we are. The gap between the vision and current state is the roadmap.

## The _try_ Problem

486 `def _try_*` functions across the reasoning module.
Each one is a hand-coded pattern matcher for a specific ARC task type.

This is the opposite of learning. Each function is ME (Claude) being smart, not THE SYSTEM being smart. Per CLAUDE.md, this is technically theater.

**However:** These functions produce the 341/1000 score. They're real results.
The honest framing: this is a hand-coded baseline that a learning system needs to beat.

## Total _try_ Counts by File

| File | _try_ count |
|------|-------------|
| arc_smart_solver.py | 636 |
| arc_solver.py | 251 |
| arc_learner.py | 64 |
| arc_multiobj.py | 52 |
| arc_advanced.py | 48 |
| arc_tiling.py | 22 |
| arc_region.py | 14 |
| arc_object_solver.py | 14 |
| arc_gravity.py | 12 |
| arc_extraction.py | 12 |

## What Needs to Happen

1. The `_try_*` functions define the ceiling for hand-coded approaches.
2. A real learning system must discover strategies that match or exceed this ceiling.
3. `brain_core/` (Rust) and `connective/` need to be wired in or deleted.
4. `deploy/push.sh` needs credentials removed.
5. Empty dirs (`workspace/`, `evals/`, `scripts/`) should be cleaned up or populated.
