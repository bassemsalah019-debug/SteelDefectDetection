# DATA_AUDIT.md
*Generated 2026-06-13 by `scripts/audit_dataset.py` + `scripts/check_leakage.py`.
Raw JSON: `docs/audit/data_audit.json`. All numbers are measured.*

## Summary verdict
The dataset is **clean and correctly split**, but it has two intrinsic properties that
**cap and distort the metric**: (1) a **tiny, high-variance test set**, and (2) an
**extreme object-scale range**. Label quality is **not** the bottleneck. Leakage is **low**.

## 1. Composition
- **1800 images, all exactly 200×200**, stored as **3-channel RGB** (grayscale content
  replicated across channels → confirms the RGB→grayscale parity fix in `src/preprocessing.py`
  is both correct and necessary).
- Split: train 1440 / val 180 / test 180; **30 images per class** in val and test (stratified).
- **4189 boxes**, mean **2.33 boxes/image** (range 1–9).

## 2. Class & instance balance
Instances per class (train): inclusion **852**, patches 688, crazing 527, rolled-in_scale 496,
scratches 427, pitted_surface **345**. → **2.5× instance imbalance** (inclusion vs pitted_surface),
even though *images* per class are balanced. Driver: inclusion images contain many small boxes;
pitted_surface few large ones. **Severity: MEDIUM** — biases the box-level loss toward inclusion.

## 3. Object scale — the structural finding
Median box **area fraction** by class (fraction of the 200×200 frame):

| class | median area | character |
|---|---|---|
| inclusion | **0.041** | tiny (multi-instance, small) |
| scratches | 0.075 | thin/elongated |
| patches | 0.093 | medium |
| rolled-in_scale | 0.123 | medium, low-contrast |
| crazing | 0.211 | large texture region |
| pitted_surface | **0.555** | covers half the frame |

Overall: median 0.118, mean 0.175, **6.3% of boxes cover >50%** and **2.8% >80%** of the image.
**Consequences (measured downstream):**
- A **13× scale gap** (inclusion 0.041 → pitted_surface 0.555) in one 6-class task → strong
  multi-scale pressure; explains why **higher resolution helps `scratches`/`inclusion`** most.
- Giant boxes make **mAP@0.5:0.95 structurally low** (~0.38 across all models) because high-IoU
  matching is unforgiving on near-full-frame boxes — this is a dataset property, not a model defect.
- **Severity: HIGH** for design implications (P2 head, resolution, anchor-free scale handling).

## 4. Label quality
- **0 missing, 0 empty, 0 invalid** labels (all coords in [0,1], class ids 0–5, 5 fields/line).
- See `docs/audit/LABEL_QUALITY_REPORT.md` for the deeper read. **Severity: NONE.**

## 5. Duplicates & leakage
- Exact (md5): **1 duplicate pair within a split, 0 across splits** → no split-bug leakage.
- Near-dup: an 8×8 dHash gave false positives (cross-class "matches" on flat textures) and was
  **discarded**. The trustworthy 32×32 pixel-correlation check (`scripts/check_leakage.py`):
  - test→train: **0** images with a twin ≥0.99 (max 0.989), 3.9% ≥0.97, ~42% of those cross-class.
  - val→train similar; val/test 30/class intact.
- **Verdict: LOW leakage.** A small tail (~4% of test) is mildly similar to a train image — expected
  for NEU-DET (crops from limited source plates) — giving a *small* optimistic bias, not invalidation.

## 6. Is data the bottleneck?
| Limiter | Bottleneck? | Impact on metric |
|---|---|---|
| Label noise | No | none (labels perfect) |
| Leakage | Marginal | small optimistic bias |
| **Test-set size (180)** | **Yes (for VALIDITY)** | ±1–2 pp seed noise → kills single-seed claims |
| **Scale range / giant boxes** | **Yes (for CEILING)** | caps mAP@.5:.95; drives resolution dependence |
| Instance imbalance | Partial | depresses minority-class (pitted_surface) recall |

**Highest-leverage data actions:** (a) **k-fold or 5-seed evaluation** to beat the 180-img
variance; (b) **scale-aware training** (P2 head / higher res) for the inclusion↔pitted_surface
13× gap; (c) **class-balanced sampling/loss** for instance imbalance. None require new labels.
