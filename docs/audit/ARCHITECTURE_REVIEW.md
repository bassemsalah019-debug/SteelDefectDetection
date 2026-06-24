# ARCHITECTURE_REVIEW.md
*Generated 2026-06-13. Measured facts are marked [M]; everything else is an explicit
[HYPOTHESIS] requiring a run before it earns a number. "Do not recommend changes based
solely on papers" is obeyed: no candidate is recommended for adoption on paper evidence — only
for **measurement**.*

## 1. The measured verdict on the current architecture choices
[M] At the fair @640 recipe (identical settings, re-measured today):

| Model | mAP@0.5 | Δ vs baseline | GFLOPs |
|---|---|---|---|
| **YOLOv8n baseline** | **0.7525** | — | 8.09 |
| LZY (Ghost+ResCBAM+WIoU) | 0.7316 | −2.1 pp | 10.17 |
| paper (Ghost+MPCA+SIoU) | 0.7305 | −2.2 pp | 6.24 |

[M] **The lightweight modifications (Ghost backbone, MPCA, SIoU, CBAM, WIoU) reduce accuracy
on this dataset.** They cost mostly `rolled-in_scale` and `inclusion` (low-contrast/small classes,
see FAILURE_ANALYSIS §3). On the RTX 2000 Ada the GFLOP saving is irrelevant — all three run
~7 ms/img [M]. **Conclusion: for this dataset + this hardware, plain YOLOv8n is the better model.**
The paper-reproduction work is valid as a *reference*, not as the production choice.

## 2. Where the real ceiling is (so we search the right space)
[M] crazing 0.44 / rolled-in_scale 0.64 are the floor; both are **low-contrast**. inclusion is
**small** (4% median area); scratches are **thin**. The bottleneck is **fine-feature / multi-scale
representation + contrast**, not detector family. This predicts which architecture changes can help.

## 3. Candidate assessment (ranked by expected value, each = a HYPOTHESIS to measure)
| Candidate | Rationale | [HYPOTHESIS] expected mAP@0.5 effect | Edge-fit (Ada) | Priority to test |
|---|---|---|---|---|
| **YOLOv8n + P2 head** | adds a stride-4 head → directly targets small `inclusion` + thin `scratches` | +0.5 to +2 pp on small classes; small/neutral overall | good (+~10% GFLOPs) | **1 (highest)** |
| **YOLOv8s** (vs n) | more channel capacity for low-contrast `crazing`/`rolled-in_scale` | +1 to +2 pp; risk of overfit on 1440 imgs | good (~28 GFLOPs, still >150 FPS) | **2** |
| **YOLO11n / YOLO12n** | newer baseline, drop-in, ~same cost; "newer arch beats paper" is a legit route | ±1 pp; free to try | excellent | **2** |
| **input 640→800/960** (recipe, not arch) | [M] already the biggest proven lever | proven +~3 pp (opt runs) | fine | **already known** |
| RT-DETR-l / RF-DETR | transformer detector; strong on COCO | [HYP] +? but **1800 imgs is too small** for DETR data-hunger; heavy | poor (edge goal) | low (research only) |
| D-FINE / DAMO-YOLO / Gold-YOLO / PP-YOLOE | modern CNN detectors | [HYP] comparable to v8/11; integration cost high (not in Ultralytics) | mixed | low |
| EfficientViT / HGNetv2 backbones | efficient attention backbones | [HYP] unclear on grayscale texture | medium | low |
| BiFPN / Dynamic Head necks | richer multi-scale fusion | [HYP] could help the 13× scale gap | good | medium (after P2) |

## 4. Loss / head components (measured + hypothesis)
- [M] SIoU and WIoU **did not help** here (they ride on the losing models). Do not adopt on this data.
- [HYPOTHESIS] Varifocal / Quality Focal Loss + DFL are worth a *controlled* ablation on the
  **baseline** (not bundled with a backbone change) — they target classification-quality ranking,
  which could help the low-AP classes. Test in isolation.

## 5. Recommendation
1. **Production model now: YOLOv8n baseline** (`results/baseline_640` / `baseline_opt`) — it is the
   measured best at the fair recipe and edge-cheap enough on Ada. Stop treating the Ghost model as "best."
2. **Highest-EV architecture experiments to actually run** (in order): **P2 head on YOLOv8n**,
   **YOLOv8s**, **YOLO11n** — each under the 5-seed protocol so the result is defensible.
3. Defer DETR-family / exotic backbones — wrong data scale and wrong deployment target; revisit only
   if a larger dataset appears.
