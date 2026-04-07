# VERIFICATION.md — How to Prove Things Work

## The Rule

No measurement = no claim.

## Quick Verification: verify.sh

```bash
./verify.sh          # Run all checks
./verify.sh --score  # Run eval and report score
./verify.sh --quick  # Skip eval, just check code health
```

## What verify.sh Checks

### Code Health (--quick)
1. **Import check** — Can `klomboagi` be imported without errors?
2. **_try_ count** — How many `_try_*` functions exist? (Trend tracking, not blocking.)
3. **Dead code detection** — Are `connective/`, `brain_core/`, `runtime/` imported by the solver?
4. **Credential scan** — Any hardcoded passwords, tokens, keys in tracked files?
5. **Test suite** — Do existing tests pass?

### Score Verification (--score)
6. **ARC eval** — Run `arc_eval.py` against full dataset.
7. **Score comparison** — Compare against last recorded score.
8. **Regression check** — Flag any tasks that previously passed but now fail.

## Before Claiming Anything

### "I added a new strategy"
- [ ] Run `./verify.sh --score`
- [ ] New score > old score
- [ ] No regressions (no previously-passing tasks now fail)
- [ ] Commit includes score in message: `[SOLVE] Description: XXX/1000 (+N)`

### "Learning is working"
- [ ] Identify the learning mechanism (code path)
- [ ] Run eval WITHOUT the learning (baseline)
- [ ] Run eval WITH the learning (treatment)
- [ ] Score difference is positive
- [ ] Ablation: remove learned state, score reverts
- [ ] No new `_try_*` functions were added

### "Infrastructure is wired in"
- [ ] Identify the entry point
- [ ] Add a log/print at the entry point
- [ ] Run the solver on one task
- [ ] Show the log output proving the code was called
- [ ] Remove the log, commit clean code

### "Runtime state persists"
- [ ] Run solver, show state file written
- [ ] Stop solver, show state file exists on disk
- [ ] Restart solver, show state file read
- [ ] Show that read state influenced a decision

### "Transfer is working"
- [ ] Define source domain and target domain
- [ ] Eval target domain without transfer (baseline)
- [ ] Train/run on source domain
- [ ] Eval target domain with transfer (treatment)
- [ ] Treatment > baseline

## Verification Log

Keep a running log of verified claims in `docs/verification_log.md`.
Format:
```
## YYYY-MM-DD — Claim
- Baseline: X
- Treatment: Y
- Delta: +/-Z
- Mechanism: description
- Verdict: CONFIRMED / FAILED / INCONCLUSIVE
```
