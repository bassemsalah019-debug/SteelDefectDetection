# REPRODUCIBILITY_REPORT.md
*Generated 2026-06-13.*

## 1. The reproducibility failure we found (and fixed)
**Incident:** the headline comparison (`results/model_comparison.txt`) was **8 days stale**. It
reported the improved model at 0.7073 while the live checkpoint (retrained 2026-06-07) scored
0.7305, and the baseline (retrained 2026-06-10) scored 0.7525 not 0.7367. A whole project
conclusion ("improved is worst at 640") rested on an un-regenerated aggregate file.

**Root cause:** aggregate report files are written by a notebook and **not regenerated when individual
runs change**. There is no single command that recomputes all metrics from checkpoints.

**Fix applied today:**
- Re-scored all @640 checkpoints with `src/eval.py` (identical settings) → `results/<run>/results.json`.
- Regenerated `results/model_comparison.txt` and `experiments/LEADERBOARD.md` from those.
- Each `results.json` records weights, data, split, imgsz, device, timestamp → self-describing.

## 2. Standing reproducibility risks
| Risk | Severity | Status |
|---|---|---|
| Single-seed claims on 180-img test | **HIGH** | `scripts/run_seed_study.py` written; **must be run** |
| Stale aggregate reports | MED | fixed; need a "recompute all" habit (see §4) |
| `lzy_640` recipe mismatch (100ep/AdamW) | MED | documented; retrain at 150ep/SGD to compare fairly |
| Recipe history drift (224→640, 100→150→200 ep) | MED | documented in model notes; current runs consistent |
| Deps unpinned for report/eval stack | LOW | fixed (`requirements.txt` pins openai/reportlab/…) |

## 3. What IS reproducible (verified)
- [M] Seeds are set and logged in every `args.yaml` (seed 42 for the saved runs).
- [M] Split is deterministic and verified: 1440/180/180, 30 img/class (`tests/test_smoke.py`).
- [M] Custom modules load via `register()`/`register_lzy()`; `src/eval.py` reproduces saved metrics
  to 4 dp → the eval path is reproducible.
- [M] Preprocessing parity is enforced by `tests/test_preprocessing_parity.py` (8 tests).

## 4. Required protocol going forward
1. **Never report a single-seed number as a comparison.** Use `run_seed_study.py` (seeds
   42/123/777/2025/3407) → mean ± std ± 95% CI; adopt a change only if the two-sample Welch test
   separates (|t|>2.78, df≈4) AND CIs are disjoint.
2. **Recompute, don't trust files.** After any retrain, re-run `src/eval.py` for the affected runs and
   regenerate `model_comparison.txt` + `LEADERBOARD.md`. (A `scripts/refresh_leaderboard.py` is the
   obvious next tool — not yet written.)
3. **One recipe for fair comparisons.** Fix {imgsz, epochs, optimizer, aug} across any models being
   compared; vary only the thing under test.
4. **Timestamp + checkpoint hash** in every reported table (results.json already carries timestamps).

## 5. Statistical-significance status
[M] **No current claim is statistically validated.** The flagship "@800 improved (0.7678) beats
baseline (0.7630)" is +0.48 pp, single-seed — **below the noise floor** of this test set. Until the
5-seed study runs, the only defensible statement is: *"baseline and improved are statistically
indistinguishable; baseline leads at the fair @640 recipe by 2.1 pp (also single-seed, to be
confirmed)."*
