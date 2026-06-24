# HYPERPARAMETER_STRATEGY.md
*Generated 2026-06-13. Levers ranked by evidence. [M]=measured on this project,
[HYP]=hypothesis to validate. Search target: TEST mAP@0.5 under the 5-seed protocol.*

## 1. What the data already tells us
- [M] **Resolution is the dominant lever.** 640→800 + TTA moved every model up ~+3 pp
  (fair @640 baseline 0.7525 → opt @800 0.7630). Bigger than any architecture change.
- [M] Inputs are **grayscale-in-RGB** → **HSV/color augmentation is wasted** (and possibly harmful);
  it should be disabled or near-zero.
- [M] Extreme scale range (4%→55% box area) → **multi-scale / scale-jitter augmentation matters**.
- [M] 180-img test, 1440 train → **regularization + EMA + early-stop** matter; big models overfit
  (YOLO11s 9.4M underperforms n at @800).

## 2. Ranked lever table
| Rank | Lever | Current | Suggested | [HYP] ΔmAP@0.5 | Confidence | Cost |
|---|---|---|---|---|---|---|
| 1 | **Input resolution** | 640 (fair) / 800 (opt) | sweep {640, 800, 960}; pick by val | +2 to +3 pp | **High [M-backed]** | high (retrain) |
| 2 | **TTA at inference** | on for opt only | standardize on (augment=True, NMS 0.6) | +1 to +2 pp | **High [M]** | ~0 (eval only) |
| 3 | **HSV aug off** (grayscale) | hsv_h/s/v default | set 0.0 | +0 to +0.5 pp, less noise | Med-High | ~0 |
| 4 | **mosaic + close_mosaic** | close_mosaic 10 | sweep close_mosaic {10,15,20}; mosaic 1.0 | +0.5 to +1 pp | Med | retrain |
| 5 | **scale / multi-scale** | scale ~0.5 | raise scale jitter; try `multi_scale=True` | +0.5 to +1 pp (small obj) | Med [HYP] | retrain |
| 6 | **copy-paste / mixup** | mixup 0.1 (opt) | copy_paste 0.1–0.3 for minority classes | +0.5 pp (pitted/rolled) | Low-Med [HYP] | retrain |
| 7 | **optimizer/LR** | SGD 0.01 + cos | keep SGD; light lr0 sweep {0.005,0.01,0.02} | ±0.5 pp | Med | retrain |
| 8 | **epochs / patience** | 150–200 / 50–60 | 200–300 + patience 50 | +0.3 pp | Med | high |
| 9 | **EMA** | on (default) | keep on | baseline hygiene | High | ~0 |
| 10 | **erasing / perspective** | low | small random-erasing for occlusion robustness | ~0–0.3 pp | Low | retrain |

## 3. Recommended search (focused, not a grid)
The single-GPU budget forbids a big grid. **Two stages:**

**Stage A — free / eval-only (do first, hours not days):**
- Standardize TTA + NMS 0.6 across all reported numbers [M-proven].
- Confirm HSV=0 has no downside (one short retrain).

**Stage B — `yolo tune` genetic search on the top config (the GPU spend):**
- Base: YOLOv8n @ imgsz 800. Search space limited to the high-EV knobs: `lr0, mosaic,
  close_mosaic, scale, copy_paste, mixup, weight_decay` (≈7 dims). 20–30 iterations.
- Then re-run the winner under the **5-seed protocol** before believing it.

## 4. Hard rule
[M] Gains of ±0.5 pp on this test set are **inside seed noise**. No HP change is "adopted" until
the 5-seed two-sample test (`scripts/run_seed_study.py --compare`) shows separation. Resolution and
TTA are the only levers currently above the noise floor.
