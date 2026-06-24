# Model Card - YOLOv8n for Steel Surface Defect Detection

> **PRODUCTION MODEL (2026-06 audit + 5-seed study + deployment benchmark): the plain
> YOLOv8n baseline @640** — TEST mAP@0.5 **0.7525** (5-seed 0.7475 ± 0.016), **157 FPS**,
> 6.3 MB, no TTA. The "improved" architectures below (Ghost+MPCA+SIoU etc.) were tested
> and did **not** beat it — they are kept as reference reproductions only. See
> `docs/audit/OPTIMIZATION_ROADMAP.md` and `docs/audit/DEPLOYMENT_BENCHMARK.md`. The
> sections below describe the improved reproduction and remain valid as such.

## Overview
- **Task:** object detection (localization + classification) of 6 surface-defect
  types on hot-rolled steel strips.
- **Architecture:** YOLOv8n with three modifications reproduced from
  *A lightweight algorithm for steel surface defect detection using improved YOLOv8*,
  Scientific Reports 2025 (s41598-025-93469-5):
  1. **Ghost backbone** (Conv -> GhostConv, C2f -> C3Ghost)
  2. **MPCA** MultiPath Coordinate Attention at the end of the backbone
  3. **SIoU** box-regression loss (replaces CIoU)
- **Size:** 2.39M params, 6.3 GFLOPs (~20% lighter than the 3.01M YOLOv8n baseline).
- **Best checkpoint:** `results/improved_opt/weights/best.pt`.

## Training data
- **NEU-DET**: 1,800 grayscale images, 200x200 px, 6 classes (300 each).
- **Split:** paper protocol 8:1:1 = 1440 train / 180 val / 180 test, stratified per
  class, seed 42. Test set is held out for the reported metric.
- **Classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

## Training configuration (improved_opt)
COCO transfer from `yolov8n.pt`; imgsz 800; 200 epochs; SGD (lr0=0.01, momentum
0.937, wd 5e-4) + cosine LR; close_mosaic 20; mixup 0.1; patience 60; seed 42;
Windows-safe `workers=0`. Evaluated with TTA (`augment=True`) + NMS IoU 0.6.

## Performance (held-out TEST set, 180 imgs)
| Metric | Value |
|---|---|
| mAP@0.5 | **0.768** (paper target 0.786) |
| mAP@0.5:0.95 | 0.384 |
| precision | 0.760 |
| recall | 0.660 |

| Class | mAP@0.5 | | Class | mAP@0.5 |
|---|---|---|---|---|
| patches | 0.951 | | scratches | 0.856 |
| pitted_surface | 0.893 | | crazing | 0.551 |
| inclusion | 0.804 | | rolled-in_scale | 0.551 |

## Limitations
- **crazing** and **rolled-in_scale** are the weakest classes (~0.55): low-contrast,
  diffuse textures with low inter-class separability. Eigen-CAM heatmaps confirm the
  model finds no localized signal to lock onto on these classes.
- Single-source dataset (one mill), grayscale, small (1,800 imgs); generalization to
  other steel lines / lighting is unverified.
- Bounding-box detection only (no pixel segmentation, no severity grading).

## Intended use
Decision-support for steel-surface inspection and as an academic reference
reproduction. Not validated for safety-critical automated rejection without
human-in-the-loop review.

## How to reproduce
1. `notebooks/01_data_preparation.ipynb` - build the dataset + split.
2. `notebooks/updated_05_train_improved.ipynb` - train (GPU).
3. `notebooks/04_evaluate.ipynb` - metrics, per-class, confusion matrix, Eigen-CAM.

## Explainability
Eigen-CAM (no gradients, principal-component projection of a late feature map) via
`src/explain.py`. See `notebooks/04_evaluate.ipynb` section 6.

## Other reproduced models (for comparison)
- Stock YOLOv8n baseline - test mAP@0.5 0.763 (opt) / 0.752 (@640).
- LZY-233 (Ghost+ResBlock_CBAM+WIoU), 4.05M - test mAP@0.5 0.732.
- YOLO11s, 9.4M - test mAP@0.5 0.755 (@800). Heavier, overfits the small set.
