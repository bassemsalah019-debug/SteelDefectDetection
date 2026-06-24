# LEADERBOARD - NEU-DET steel surface defect detection

One row per real run, **TEST** split (the paper's metric). Every number traces to a saved
file. The @640 rows were **re-measured 2026-06-13** with identical settings via `src/eval.py`
(→ `results/<run>/results.json`); the @800 "opt" rows come from each run's `metrics_summary.txt`.

- **Split:** stratified 8:1:1 = 1440 / 180 / 180 (`configs/neu_det.yaml`).
- **Device:** RTX 2000 Ada (16 GB). GFLOPs@640 measured by `ultralytics.get_flops`.
- ⚠ **All rows are single-seed; the 180-img test set is high-variance.** Treat any gap
  < ~1.5 pp as a tie until the 5-seed protocol (`scripts/run_seed_study.py`) is run.

## Fair recipe — imgsz 640, identical eval (re-measured 2026-06-13)
| # | Run | Model | TEST mAP@0.5 | mAP@.5:.95 | P | R | params | GFLOPs@640 |
|---|---|---|---|---|---|---|---|---|
| 1 | `baseline_640` | **YOLOv8n baseline** | **0.7525** | 0.3926 | 0.743 | 0.665 | 3.01M | 8.09 |
| 2 | `lzy_640`* | Ghost+ResCBAM+WIoU | 0.7316 | 0.3685 | 0.641 | 0.732 | 4.05M | 10.17 |
| 3 | `improved_640` | Ghost+MPCA+SIoU (paper) | 0.7305 | 0.3815 | 0.664 | 0.700 | 2.39M | 6.24 |

\* `lzy_640` used a different recipe (100 epochs, AdamW) — not recipe-matched.

**At the fair recipe the plain baseline wins by ~2.1 pp.** The architecture modifications
reduce accuracy (mostly `rolled-in_scale`, `inclusion`) for GFLOPs savings that are
irrelevant on the Ada target. → see `docs/audit/ARCHITECTURE_REVIEW.md`.

## Optimized recipe — imgsz 800, 200 epochs, TTA + NMS 0.6 (from metrics_summary.txt)
| # | Run | Model | TEST mAP@0.5 | mAP@.5:.95 | P | R |
|---|---|---|---|---|---|---|
| 1 | `improved_opt` | Ghost+MPCA+SIoU | 0.7678 | 0.3841 | 0.760 | 0.660 |
| 2 | `baseline_opt` | YOLOv8n baseline | 0.7630 | 0.3817 | 0.727 | 0.707 |
| 3 | `lzy_opt` | Ghost+ResCBAM+WIoU | 0.7554 | 0.3776 | 0.716 | 0.703 |
| 4 | `yolo11s_opt` | YOLO11s | 0.7546 | 0.3905 | 0.707 | 0.711 |
| 5 | `yolo11s_960` | YOLO11s @960 | 0.7457 | 0.3885 | 0.684 | 0.703 |

At @800+TTA `improved_opt` (0.7678) leads `baseline_opt` (0.7630) by **+0.48 pp** — almost
certainly inside the noise band of a 180-img test set (hence the seed protocol).

## Reference rows (literature — NOT our measurements)
| Source | Model | mAP@0.5 | Params | GFLOPs |
|---|---|---|---|---|
| Ma et al. 2025 (paper) | Improved YOLOv8 | 0.786 | 2.04M | 5.1 |
| Ma et al. 2025 (paper) | YOLOv8n baseline | 0.774 | — | — |
| LZY-233 (GitHub) | Ghost+ResCBAM+WIoU | 0.792 | 4.05M | 10.2 |

## Per-class TEST mAP@0.5 @640 (re-measured)
| model | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| baseline | 0.440 | 0.858 | 0.925 | 0.862 | 0.636 | 0.794 |
| paper | 0.416 | 0.842 | 0.934 | 0.844 | 0.574 | 0.773 |
| LZY | 0.444 | 0.804 | 0.935 | 0.869 | 0.521 | 0.818 |

`crazing` (~0.42–0.44) is the floor for every model; `rolled-in_scale` is second-weakest.

## 5-SEED VALIDATED RESULT (2026-06-14) — the statistical gate
Seeds 42/123/777/2025/3407, locked grayscale recipe @640, scored on TEST. This **supersedes
all single-seed @640 rows above** for comparison purposes.

| Model | mean ± std | 95% CI | vs baseline | Welch t | significant? |
|---|---|---|---|---|---|
| **YOLOv8n baseline** | **0.7475 ± 0.0161** | [0.728, 0.767] | — | — | — |
| + P2 head | 0.7239 ± 0.0220 | [0.697, 0.751] | −2.4 pp | −1.93 | **No** (trends worse) |
| YOLOv8s (n=3) | 0.7338 ± 0.0071 | [0.716, 0.751] | −1.4 pp | −1.65 | **No** (trends worse) |

**Conclusion: no candidate beats plain YOLOv8n.** The baseline std (±1.6 pp; seeds span
0.721–0.764) proves single-seed gaps < ~3 pp are noise — retiring the old "+0.48 pp improved
wins" claim. **Decision: ship the YOLOv8n baseline.** (`experiments/seed_study/`,
`scripts/run_seed_study.py`.)

## Status vs the paper
Best honest TEST mAP@0.5 = **0.7678** (`improved_opt`, @800+TTA, single seed), **−1.8 pp**
below the paper's 0.786. At the fair @640 recipe nothing beats the **baseline's 0.7525**.
Not yet satisfied: beat 0.786; mean±std over ≥3 seeds; measured FPS at GFLOPs ≤ ~5.

## Discrepancy — RESOLVED (2026-06-13)
The improved model's 0.7073 (old `model_comparison.txt`, 2026-06-05) vs 0.7305
(`improved_640/metrics_summary.txt`, 2026-06-07): **0.7073 was stale** — it predated a
2026-06-07 retrain. Re-evaluation today reproduces **0.7305** exactly (harness validated).
`model_comparison.txt` has been regenerated from current checkpoints. Full trace:
`docs/audit/EVALUATION_AUDIT.md`.
