# Project Audit & Completion Plan

A full read of every file in the repo, the design decisions for the deployment
layer, and a cleanup/refactor plan. Analysis only - nothing here is auto-deleted.

> **2026-06-13 DEEP AUDIT (supersedes the metric tables below).** A full evidence-first
> re-audit was run. Headline change: re-measured with identical settings, the **plain
> YOLOv8n baseline leads the fair @640 comparison (0.7525)** — the Ghost/MPCA/SIoU and
> CBAM/WIoU modifications *reduce* accuracy on this dataset. The old 0.7073 was a stale
> file (resolved). See **`docs/audit/`**: `EVALUATION_AUDIT.md`, `DATA_AUDIT.md`,
> `LABEL_QUALITY_REPORT.md`, `FAILURE_ANALYSIS.md`, `ARCHITECTURE_REVIEW.md`,
> `HYPERPARAMETER_STRATEGY.md`, `DEPLOYMENT_RECOMMENDATIONS.md`, `REPRODUCIBILITY_REPORT.md`,
> and **`OPTIMIZATION_ROADMAP.md`** (the ranked plan). Authoritative metrics now live in
> `experiments/LEADERBOARD.md` + each `results/<run>/results.json`.

## 1. What the project is
Reproduce and fairly compare lightweight YOLOv8 variants for steel-surface defect
detection on NEU-DET (6 classes, 1,800 imgs, paper 8:1:1 split). Best model:
**improved YOLOv8n (Ghost+MPCA+SIoU)**, `results/improved_opt/weights/best.pt`,
test mAP@0.5 = 0.768 at 2.39M params.

## 2. Results (held-out TEST set)
| Model | Params | fair @640 | opt @800 | repo/paper target |
|---|---|---|---|---|
| baseline YOLOv8n | 3.01M | 0.737 | 0.763 | 0.774 |
| **improved (Ghost+MPCA+SIoU)** | 2.39M | 0.707 | **0.768** | 0.786 |
| LZY (Ghost+ResCBAM+WIoU) | 4.05M | 0.732 | - | 0.792 |
| YOLO11s | 9.4M | - | 0.755 (@800) | - |

**Honest nuance:** at the fair 640 recipe the improved model *loses* to baseline; it
only wins after the heavier 800px + TTA recipe. The Ghost backbone needs resolution
to pay off. crazing / rolled-in_scale (~0.55) are the floor across all models.

> **Correction (2026-06-10 audit):** the @640 cells above do not match the saved
> `results/<run>/metrics_summary.txt`: `improved_640` actually saved **0.7305** (not
> 0.707), and `baseline_640` has **no** saved TEST eval (so the 0.737 here is
> unverified). The "improved loses to baseline at 640" claim is therefore not currently
> backed by a saved baseline@640 TEST metric. **`experiments/LEADERBOARD.md` (built
> only from saved files) is now the authoritative metric source;** reconcile/re-measure
> these @640 figures before citing them.

## 3. Where current SOTA sits (2024-2026)
Lightweight YOLO: 0.77-0.80. Transformer/hybrid (SH-DETR 0.83, MDT-Net 0.827,
HCT-Det 0.795, DEENet 0.814) lead but at multiples of the compute - not edge-viable.
Keeping the lightweight YOLO direction is the right call for a deployment project.

## 4. Defects found and fixed
- `src/app.py` pointed at a non-existent `results/baseline/...` and never called
  `register()`, so it could not load the best (custom) model. **Rewritten.**
- Missing `04_evaluate.ipynb` (promised in README). **Created.**
- No XAI, no export, no deployment files. **Added** (`src/explain.py`,
  `src/export_model.py`, `deployment/huggingface/`, `docs/DEPLOYMENT.md`).
- `notebooks/updated_06_train_lzy.ipynb` referenced by docs but missing (its results
  `results/lzy_640/` exist). **Recommend recreating** from `configs/yolov8n_lzy.yaml`
  + `register_lzy()` for full reproducibility.

## 5. Cleanup plan (do via git after `git init`; nothing auto-removed)
| File / dir | Action | Reason |
|---|---|---|
| `data/neu-det-yolo/labels/*.pt` | remove | weights wrongly saved into the labels dir |
| `results/improved_640/` | remove | stale partial run (3 jpgs, no weights) |
| `notebooks/_tune_aug.py` | remove | one-shot script, already applied; references missing nb06 |
| `notebooks/*.pt` (dup weights) | move to `models/pretrained/` | duplicated at root |
| `src/voc_to_yolo.py` | merge/keep | redundant with nb01 + prepare_local |
| `src/prepare_local.py` | keep/document | builds 2-way split; superseded by 3-way |
| `configs/neu_det.yaml` | reconcile | notebooks use the generated `data/.../data.yaml` |
| `results/README.txt` | refresh | stale ("notebooks 03 and 04") |

## 6. Roadmap status
- [x] Phase 1 partial - audit + bug fixes (cleanup deletions left for you to git-commit)
- [ ] Phase 2 - train YOLOv11n on the identical recipe (GPU; run in the notebook)
- [x] Phase 3 - Explainable AI (Eigen-CAM in `src/explain.py` + nb 04 section 6)
- [x] Phase 4 - export/optimization (`src/export_model.py`, ONNX hub + parity)
- [x] Phase 5 - Streamlit app (`src/app.py`: upload/webcam + XAI)
- [x] Phase 6 - Hugging Face Space (`deployment/huggingface/`, `docs/DEPLOYMENT.md`)
- [x] Phase 7 - mobile/edge export path + FPS table (`docs/DEPLOYMENT.md`)
- [x] Phase 8 - final evaluation + model card (`docs/model_card.md`, nb 04)

## 7. Final recommendation
Keep the improved YOLOv8n as the deployed model. Optional single experiment:
YOLOv11**n** (not s) on the exact recipe - adopt only if it beats 0.768 on TEST.
Do not switch to a transformer: +4-6 pp for 5-10x compute kills the camera/mobile goals.
