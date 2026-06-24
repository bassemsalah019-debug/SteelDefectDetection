# OPTIMIZATION_ROADMAP.md
*Generated 2026-06-13. The ranked, evidence-based plan. Sorted by expected real-world impact
(validity first, then measured-lever gains, then speculative gains). [M]=measured, [HYP]=hypothesis.*

> **UPDATE 2026-06-14 — the 5-seed gate (#1) has RUN.** Validated @640: baseline
> **0.7475 ± 0.0161**, P2 head 0.7239 ± 0.022 (−2.4 pp, not significant), YOLOv8s 0.7338
> (−1.4 pp, not significant). **No candidate beats the baseline.** Roadmap items #1 done;
> #3 (P2) and #4 (YOLOv8s) **tested and rejected**. Remaining upside is recipe (#2 @800+TTA),
> contrast (#6), and deployment (#8) — not architecture. Decision: **ship YOLOv8n baseline.**

## TL;DR — the three things this audit changed
1. [M] **The baseline YOLOv8n is the best model on this dataset** (fair @640: 0.7525 vs ≤0.7316 for
   the "improved" architectures). The Ghost/MPCA/SIoU/CBAM/WIoU work is a valid *reference*, not the
   production model. **Stop reporting the Ghost model as "best."**
2. [M] **The 0.7073↔0.7305 discrepancy was a stale file**; resolved, artifacts regenerated.
3. [M] **No comparison is statistically valid yet** (single seed, 180-img test). Fix this before
   chasing any gain.

---

## Ranked roadmap

| # | Action | Why / evidence | Exp. ΔmAP@0.5 | Confidence | Eng. effort | Compute | Risk |
|---|---|---|---|---|---|---|---|
| 0 | **Adopt YOLOv8n baseline as the production/deployed model** | [M] leads fair @640 by +2.1 pp; edge-cheap on Ada | n/a (correctness) | **High** | trivial (config) | none | none |
| 1 | **Run the 5-seed protocol** (`run_seed_study.py`) for baseline + top candidate | [M] all claims single-seed on 180 imgs; +0.48 pp gaps are noise | n/a (unlocks valid claims) | **High** | low (script ready) | ~10 trainings | low |
| 2 | **Standardize resolution 800 + TTA + NMS 0.6** as the reported recipe | [M] biggest proven lever (+~3 pp vs fair @640) | **+2 to +3** | **High** | low | retrain/eval | low |
| 3 | **P2 small-object head on YOLOv8n** | [M] 13× scale gap; `inclusion` 4% area, thin `scratches` | +0.5 to +2 (small classes) | Med-High | medium (YAML) | 5 trainings | low |
| 4 | **YOLOv8s and YOLO11n vs n** (capacity / newer base) | [M] low-contrast classes look capacity-limited; YOLO11s overfits but s/11n untested at this recipe | +1 to +2 | Medium | low | 10 trainings | med (overfit) |
| 5 | **HSV aug → 0 + mosaic/scale tuning** | [M] inputs are grayscale → color aug wasted; scale jitter targets multi-scale | +0 to +1 | Medium | low | retrain | low |
| 6 | **CLAHE / adaptive-contrast preprocessing** (inside the parity contract) | [M] `crazing` 0.44 & `rolled-in_scale` 0.64 are low-contrast floors | +0.5 to +2 on weak classes | Medium | medium | retrain | **med** (must keep train/infer parity) |
| 7 | **`yolo tune` genetic search** on the top config (≈7 dims, 20–30 iters) | standard last-mile HPO | +0.3 to +1 | Medium | low | many trainings | low |
| 8 | **TensorRT FP16 engine + benchmark** | [M] Ada target; >100 FPS already, FP16 safe | n/a (deploy) | High | medium | GPU build | low |
| 9 | **Class-balanced sampling / copy-paste for minority classes** | [M] 2.5× instance imbalance (inclusion vs pitted) | +0 to +0.5 | Low-Med | medium | retrain | low |
| 10 | **INT8 quantization** (only if throughput needed) | multi-stream headroom | negative or 0 | n/a | medium | GPU+calib | **med** (weak classes degrade) |

### Validity-vs-performance split
- **Do #0 and #1 first.** They cost little and make every later number trustworthy. Skipping them
  means re-litigating noise forever (which is how the project ended up with a stale headline).
- **#2 is the proven win.** It's already partly done (the "opt" runs) — formalize it.
- **#3–#7 are the genuine research upside.** Each is a *hypothesis* until the 5-seed gate passes it.

## Highest-confidence path to the best model (the answer to "what should we actually do")
1. Lock recipe = YOLOv8n, imgsz 800, SGD+cos, HSV 0, mosaic+close_mosaic 20, TTA eval.
2. Add a P2 head; train baseline-P2 and plain baseline across 5 seeds.
3. If P2 wins the two-sample test → that's the new model. If not → plain baseline @800 is the honest best.
4. One `yolo tune` pass on the winner; re-validate over seeds.
5. Export winner to TensorRT FP16; benchmark; ship.

**Honest expectation:** the defensible best is likely **~0.76–0.78 mAP@0.5** (baseline @800 ± P2/HPO),
i.e. **around or just under the paper's 0.786** — and that is the *truthful* ceiling for this 1800-image,
low-contrast dataset. The biggest remaining lever is **not architecture or HPO — it is more/better
data and the contrast problem on `crazing`/`rolled-in_scale`**, which no detector tweak will fully solve.

## What this audit did NOT do (needs your GPU / a gate)
- Run the 5-seed trainings, the P2/v8s/11n experiments, `yolo tune`, or the TensorRT build — all are
  **prepared** (`scripts/run_seed_study.py`, configs) but are GPU jobs you launch.
- Numbers in the [HYP] columns are **estimates, not measurements**, and are labeled as such throughout.
