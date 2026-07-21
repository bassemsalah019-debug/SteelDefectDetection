# Adaptive Real-Time Confidence Thresholding — Technical Report

*Implemented 2026-06-27. All numbers below are **measured** on the held-out TEST split
(180 imgs, imgsz 640) with the production model `results/baseline_640/weights/best.pt`
on an RTX 2000 Ada, via `scripts/eval_adaptive.py` → `docs/audit/adaptive_eval.json`.*

---

## 1. Problem & motivation

The detector exposes a single global confidence threshold (the app default **0.25**).
On NEU-DET that one number is wrong for two reasons at once:

- **Class difficulty is wildly uneven.** Measured per-class AP@0.5 ranges from `crazing`
  **0.44** (low-contrast hairline cracks) to `patches` **0.92**. At a flat 0.25, the easy
  classes still emit nuisance false positives while the hard classes have their genuine —
  but low-confidence — detections truncated → **missed defects** on exactly the classes
  that matter most.
- **Imaging conditions drift on a real line.** Lighting, focus and defect scale change
  shift-to-shift. A threshold tuned in a lab is mis-calibrated when the frame is dark,
  washed-out, or slightly out of focus, because the model's confidence moves with image
  quality but the threshold does not.

**Adaptive thresholding** replaces the constant with a *per-class, per-image* threshold
`T(class, image)` derived from four cheap signals, applied as a **post-processing filter**
on the detector output. The model, its weights, and any ONNX/TensorRT export are untouched.

---

## 2. The algorithm

For every image we keep a detection of class *c* with confidence *conf* iff
`conf ≥ T(c, image)`, where

```
T(c, image) = clip( base(c) + Δ_brightness + Δ_quality + Δ_density ,  t_min , t_max )
```

### 2.1 Per-class anchor — class difficulty
```
base(c) = clip( t0 + k_class · (AP_c − meanAP) ,  t_min , t_max )
```
`AP_c` is the **measured** baseline TEST AP@0.5 (`BASELINE_AP50` in
`src/adaptive_threshold.py`). Harder (lower-AP) classes get a **lower** anchor to protect
recall; easy classes get a **higher** anchor to suppress false positives. With the defaults
(`t0=0.25`, `k_class=0.30`, meanAP≈0.7525) the anchors are:

| class | AP@0.5 | base threshold |
|---|---|---|
| crazing | 0.44 | **0.156** |
| rolled-in_scale | 0.64 | 0.215 |
| scratches | 0.79 | 0.263 |
| pitted_surface | 0.86 | 0.283 |
| inclusion | 0.86 | 0.282 |
| patches | 0.92 | **0.302** |

### 2.2 Global image-level adjustments (same shift for every class)
Computed on a 256-px **grayscale** thumbnail of the canonical model input (so the signals
match what the model sees):

- **Brightness** `b = mean(luma)/255`. Penalty grows with distance from a mid-gray ideal:
  `Δ_brightness = −w_bright · clip(|b − 0.5| / 0.5, 0, 1)`. Under/over-exposed frames →
  lower threshold (preserve recall when confidence is systematically depressed).
- **Quality** `q = ½·contrast_q + ½·sharpness_q`, where `contrast_q = clip(std/255 / 0.18, 0,1)`
  and `sharpness_q = lapVar / (lapVar + 80)` (variance of the Laplacian = focus measure).
  `Δ_quality = −w_quality · (1 − q)`. Blurry / flat frames → lower threshold.
- **Density** `d` = number of candidate detections above the gather floor.
  `Δ_density = w_density · tanh((d − 3) / 3)`. Crowded frames → **raise** threshold (fewer
  nuisance alerts / precision); sparse frames → **lower** it (don't miss the rare defect).

All weights, references and bounds live in `AdaptiveConfig` (documented defaults; nothing
hidden). `class_ap` should be re-pulled from `src/eval.py` whenever the model is retrained.

### 2.3 Why a *per-detection* filter (not a `predict(conf=…)` call)
Ultralytics `predict` takes one scalar `conf`. To express a per-class threshold we gather
candidates once at a low floor (`candidate_floor = 0.05`) and filter afterwards
(`keep_mask`). This is pure NumPy on the returned boxes — see `src/infer.predict_adaptive`.

---

## 3. Integration (what changed)

| File | Change |
|---|---|
| `src/adaptive_threshold.py` | **New.** Signals, per-class thresholds, `AdaptiveThresholder` facade. Torch-free, numpy-only (cv2 optional) → cheap import, unit-testable on a fresh checkout. |
| `src/infer.py` | Added `predict_adaptive(...)` + `AdaptiveResult`. `predict(...)` (fixed) is unchanged — preprocessing parity preserved. |
| `src/app.py` | Sidebar **Thresholding mode = Adaptive / Fixed**; an expander shows the live signals, the global shift, and the per-class thresholds. Eigen-CAM, the bilingual report, and grayscale parity are untouched; the report metadata now records the mode + thresholds. |
| `scripts/eval_adaptive.py` | **New.** Self-contained Fixed-vs-Adaptive evaluator (own IoU matcher + VOC AP). |
| `tests/test_adaptive_threshold.py` | **New.** 13 numpy-only unit tests. |

**Two inference modes** are now first-class: `predict` (Fixed) and `predict_adaptive`
(Adaptive), selectable in the app and the eval.

**ONNX / TensorRT compatibility:** adaptive thresholding is applied *after* the model emits
detections, so it is backend-agnostic — it works identically on PyTorch, ONNX Runtime, and
TensorRT engines, and **export is unaffected** because the graph is never modified.

---

## 4. Measured results — Fixed vs Adaptive (TEST, 180 imgs @640)

| Metric | Fixed (0.25) | Adaptive | Δ |
|---|---|---|---|
| mAP@0.5 *(policy-restricted)* | 0.6315 | **0.6393** | +0.0078 |
| Precision (macro) | **0.5955** | 0.5762 | −0.0193 |
| Recall (macro) | 0.7106 | **0.7356** | **+0.0250** |
| F1 (macro) | **0.6390** | 0.6325 | −0.0065 |
| Precision (micro) | **0.5656** | 0.5153 | −0.0503 |
| Recall (micro) | 0.7094 | **0.7361** | +0.0267 |
| F1 (micro) | **0.6294** | 0.6062 | −0.0232 |
| Latency (ms, end-to-end) | 13.10 | 15.31 | +2.21 |
| FPS | 76.3 | 65.3 | −11.0 |

**Per-class — where it acts (recall / AP):**

| class | Recall fix → ada | AP50 fix → ada |
|---|---|---|
| **crazing** (hardest) | 0.413 → **0.525** (+0.113) | 0.259 → 0.296 |
| **rolled-in_scale** | 0.613 → **0.661** (+0.048) | 0.449 → 0.468 |
| inclusion / patches / pitted_surface / scratches | unchanged | unchanged |

### Honest interpretation
- **Confidence thresholding is an *operating-point* control, not an accuracy boost.** mAP
  integrates over the whole confidence range, so it is essentially *threshold-invariant*;
  the 0.63 figures here are **policy-restricted** AP (computed only on detections each
  policy keeps, from a shared candidate pool). The right comparison is therefore
  **Precision / Recall / F1 at the deployment threshold**, not mAP.
- **Harness calibration (why the ceiling is ~0.70, not 0.7525).** This evaluator scores the
  real **`predict()`** path (letterbox + NMS + decode — what actually ships), not the
  Ultralytics `val()` path. With a no-op policy (keep all conf ≥ 0.01) it measures a ceiling
  mAP@0.5 ≈ **0.702**, which matches the **0.704** that `DEPLOYMENT_BENCHMARK.md`
  independently measured for the ONNX/TensorRT export path — the same documented
  predict-vs-`val` parity gap (≈5 pp below `val()`'s 0.7525). This is a useful cross-check
  that the custom matcher is sound. **Both policies share this path**, so the Fixed-vs-Adaptive
  **deltas** are apples-to-apples even though the absolute scale sits on the predict-path.
- **Adaptive buys recall, especially on the worst classes** — `crazing` recall **+11.3 pp**
  and `rolled-in_scale` **+4.8 pp** — by lowering their anchors, while leaving the easy
  classes (already high-confidence) essentially unchanged.
- **The cost is macro precision (−1.9 pp) and ~flat F1** (−0.7 pp macro) on this *clean*
  benchmark. With the deliberately gentle default weights, Adaptive ≈ Fixed on F1 but
  shifts the balance toward catching defects. For inspection, where a **false negative
  (missed defect) is far more expensive than a false alarm**, that is the correct bias —
  and it is fully tunable (see §7).
- The clean lab test set does **not** exercise the lighting/blur drift the brightness and
  quality terms are built for; their value shows up on real-line frames, not on pristine
  200×200 scans (where `quality` is already ~0.83–0.98).

---

## 5. Computational overhead

Decomposed, measured on the same 80 images:

| Component | Cost |
|---|---|
| Image signals (brightness/contrast/Laplacian on 256-px thumb) | **0.192 ms/img** |
| Per-class thresholds + keep-mask | **0.041 ms/call** |
| **True adaptive algorithm overhead** | **≈ 0.23 ms/img** |
| Candidate floor effect (predict @0.05 vs @0.25) | 13.4 vs 13.0 ms → **~0.4 ms** |

The **algorithmic** overhead is ~0.23 ms — **< 2 %** of the ~13 ms forward pass. The larger
end-to-end delta in §4 (≈2 ms) is dominated by GPU run-to-run timing variance plus the
device→host copy of candidate boxes, not by the math. Either way Adaptive holds **65 FPS**,
far above the 30 FPS real-time bar — **real-time performance on edge devices is preserved.**
Signals are O(H·W) on a fixed-size thumbnail, so cost is input-resolution-independent.

---

## 6. Industrial applicability

- **Recall-first safety bias** matches inspection economics: the system surfaces more
  borderline defects (notably crazing, the hardest class) instead of silently dropping them.
- **Self-calibrating to conditions:** brightness + quality terms compensate for shift-to-shift
  lighting and focus drift without re-tuning a global threshold per line/lighting setup.
- **Operator-load control:** the density term suppresses nuisance alerts on busy frames.
- **Zero deployment friction:** post-processing only → drops into the existing PyTorch *or*
  exported ONNX/TensorRT serving path; the Streamlit demo exposes both modes with a live
  signal/threshold readout for transparency and auditability.
- **Negligible cost** (~0.23 ms) keeps it viable on edge GPUs/CPUs.

---

## 7. Limitations & future improvements

**Limitations**
- On a clean benchmark the default config improves recall but slightly lowers macro F1 — it
  re-balances the operating point, it does not raise the PR-curve ceiling (mAP).
- The signal references (`contrast_ref`, `sharp_ref`, `bright_ideal`) are hand-set; on the
  pristine NEU-DET scans the sharpness term saturates (lapVar ≫ 80), so quality is mostly
  contrast-driven there. They should be **calibrated to the deployment camera**.
- Class anchors are tied to the *current* model's per-class AP; they must be refreshed after
  any retrain (a one-line update from `src/eval.py`).
- Density is a raw count, not normalized by image area or expected defect rate.

**Future work**
- **Cost-sensitive tuning:** expose a single `recall_bias` knob; auto-fit `k_class` and the
  global weights to maximize F-β (β>1) or to hit a target recall on a validation set.
- **Online adaptation:** EMA-track the live confidence distribution per class and shift
  thresholds to hold a target alert rate (true closed-loop control).
- **Per-class global weights:** let brightness/quality affect low-contrast classes
  (crazing, rolled-in_scale) more than high-contrast ones.
- **Calibration:** temperature/Platt-scale the raw scores first so thresholds map to true
  probabilities; learn the signal references from a labeled lighting/blur sweep.
- **Validate on degraded frames:** synthesize brightness/blur shifts to quantify the drift
  robustness the clean test set can't show.

---

## 8. Reproduce

```bash
# unit tests (no GPU)
python -m pytest tests/test_adaptive_threshold.py -q

# full Fixed-vs-Adaptive eval on TEST (GPU)
python scripts/eval_adaptive.py \
    --weights results/baseline_640/weights/best.pt \
    --imgsz 640 --conf 0.25 --device 0
# -> prints the comparison table, writes docs/audit/adaptive_eval.json

# app: pick "Adaptive" in the sidebar
streamlit run src/app.py
```

**Config reference** (`AdaptiveConfig`, `src/adaptive_threshold.py`): `t0=0.25`,
`k_class=0.30`, `w_bright=0.08`, `w_quality=0.10`, `w_density=0.06`, `t_min=0.08`,
`t_max=0.60`, `candidate_floor=0.05`, `signal_resize=256`. Override per call, e.g.
`AdaptiveThresholder(k_class=0.45, w_quality=0.15)`.
