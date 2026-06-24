# EVALUATION_AUDIT.md
*Generated 2026-06-13. Every number here was measured today or traced to a saved file.*

## Mandate
"If evaluation is flawed, stop and fix evaluation before recommending model changes."
This audit did exactly that, and it changed the project's headline conclusion.

## 1. The 0.7073 vs 0.7305 discrepancy — RESOLVED
| Artifact | improved @640 mAP@0.5 | mtime | Status |
|---|---|---|---|
| `results/model_comparison.txt` | 0.7073 | 2026-06-05 20:50 | **STALE** |
| `results/improved_640/results.csv` | (150-epoch retrain, val 0.727) | 2026-06-07 22:09 | current |
| `results/improved_640/metrics_summary.txt` | 0.7305 | 2026-06-07 22:10 | current |
| **Re-eval today (`src/eval.py`)** | **0.7305** | 2026-06-13 | **authoritative** |

**Root cause:** `model_comparison.txt` was generated 2026-06-05 and never regenerated after
the improved model was retrained on 2026-06-07. The 0.7073 is a pre-retrain number.
Re-running the eval harness today reproduces **0.7305 to 4 dp**, which simultaneously
(a) confirms 0.7305 is correct and (b) **validates `src/eval.py`** against the notebook eval.

## 2. The "fair comparison" was never controlled
Reading each run's `args.yaml`:
| Run | epochs | optimizer | imgsz | retrained | 
|---|---|---|---|---|
| baseline_640 | 150 | SGD | 640 | 2026-06-10 |
| improved_640 | 150 | SGD | 640 | 2026-06-07 |
| lzy_640 | **100** | **auto (AdamW)** | 640 | 2026-06-04 |

LZY used a different optimizer and 2/3 the epoch budget — its row is **not recipe-matched**.
And `model_comparison.txt` (2026-06-05) predated the baseline (06-10) and improved (06-07)
retrains, so **all three of its rows were stale or confounded.**

## 3. Corrected, identical-settings @640 TEST comparison (measured 2026-06-13)
Settings: `imgsz=640, split=test, no TTA, default conf/iou, cuda:0`. → `results/<run>/results.json`.

| Model | params | GFLOPs | **mAP@0.5** | mAP@.5:.95 | P | R |
|---|---|---|---|---|---|---|
| **baseline YOLOv8n** | 3.01M | 8.09 | **0.7525** | 0.3926 | 0.743 | 0.665 |
| LZY (Ghost+ResCBAM+WIoU)* | 4.05M | 10.17 | 0.7316 | 0.3685 | 0.641 | 0.732 |
| paper (Ghost+MPCA+SIoU) | 2.39M | 6.24 | 0.7305 | 0.3815 | 0.664 | 0.700 |

\* different recipe. **The baseline leads by ~2.1 pp** — the opposite of the prior conclusion.

## 4. Eval-setting hygiene
- Splits verified: 1440/180/180, 30 img/class in val & test (`tests/test_smoke.py`, re-confirmed).
- The "opt" runs evaluate with **TTA (`augment=True`) + NMS IoU 0.6**; the fair @640 runs do not.
  TTA + the resolution jump (640→800) is the single biggest measured driver of the @800 gains —
  **a recipe effect, not architecture.**
- No fixed confidence threshold is baked into mAP (correct — mAP sweeps confidence).

## 5. Residual evaluation risks (severity)
| Risk | Severity | Evidence | Fix |
|---|---|---|---|
| Single-seed comparisons on 180-img test | **HIGH** | @800 improved-vs-baseline gap = +0.48 pp | 5-seed protocol (`scripts/run_seed_study.py`) |
| Stale aggregate files | **MED** (now fixed) | `model_comparison.txt` was 8 days stale | regenerate from `results.json` on every change |
| `lzy_640` not recipe-matched | MED | args.yaml | retrain at 150ep/SGD before any LZY claim |
| Mild near-dup tail (~4% test) | LOW | `scripts/check_leakage.py`: 0 twins ≥0.99 | acceptable; documented |

## Verdict
The evaluation pipeline itself is **sound** (correct splits, correct metric, harness
reproduces saved numbers). The flaw was **stale/ confounded aggregate reporting**, now
corrected. **Before any architecture claim, the 5-seed protocol must run** — the only
remaining evaluation-validity gap.
