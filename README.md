# Steel Surface Defect Detection with YOLOv8

Graduation capstone project — automated detection of surface defects on hot-rolled
steel strips using the **NEU-DET** dataset and **YOLOv8**.

## Project Goal

Replace slow, error-prone manual visual inspection of steel surfaces with an
automated deep-learning detector that locates and classifies six defect types in
real time.

## The 6 Defect Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | crazing | Network of fine surface cracks |
| 1 | inclusion | Foreign material embedded in the surface |
| 2 | patches | Localized irregular surface patches |
| 3 | pitted_surface | Small pits / cavities |
| 4 | rolled-in_scale | Oxide scale pressed into the surface during rolling |
| 5 | scratches | Linear mechanical scratches |

## Dataset: NEU-DET

- **1,800 images** total — 300 per class × 6 classes
- Grayscale, 200×200 pixels
- Source: Northeastern University (NEU)
- Annotations: bounding boxes (originally Pascal VOC XML, converted to YOLO txt)

**Download options:**
1. Auto-download via `kagglehub` (handled inside `01_data_preparation.ipynb`)
2. Manual: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
3. Pre-converted YOLO format: https://github.com/Marfbin/NEU-DET-with-yolov8

## Project Structure

```
SteelDefectDetection/
├── notebooks/
│   ├── 01_data_preparation.ipynb        # convert VOC→YOLO + 8:1:1 split
│   ├── 02_eda.ipynb                     # class balance, samples, t-SNE, heatmaps
│   ├── 03_train_baseline.ipynb          # YOLOv8n baseline @640
│   ├── updated_03_train_baseline.ipynb  # optimized baseline @800 + TTA
│   ├── updated_05_train_improved.ipynb  # Ghost+MPCA+SIoU (best model)
│   ├── updated_07_train_yolo11s.ipynb   # YOLO11s experiment
│   └── 04_evaluate.ipynb                # metrics, confusion matrix, predictions, Eigen-CAM
├── src/
│   ├── modules/                         # MPCA, SIoU, ResBlock_CBAM, WIoU + register()
│   ├── explain.py                       # Eigen-CAM explainability (no extra deps)
│   ├── app.py                           # Streamlit app (upload/webcam → detect + XAI)
│   ├── export_model.py                  # ONNX/TFLite/NCNN export + parity check
│   └── make_paper_split.py              # 8:1:1 stratified re-split
├── configs/                            # dataset + improved/LZY architecture YAMLs
├── deployment/huggingface/             # Hugging Face Space (app, requirements, card)
├── data/                               # dataset (auto-created; git-ignored)
├── results/                            # training runs + plots (git-ignored)
├── docs/                               # presentation, model_card.md, DEPLOYMENT.md, notes
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Activate your existing venv (torch 2.6.0+cu124 already installed)
#    Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 2. Install project requirements (does NOT touch torch)
pip install -r requirements.txt
```

## How to Run (in order)

1. **`notebooks/01_data_preparation.ipynb`** — converts annotations to YOLO format,
   creates the 8:1:1 train/val/test split, writes the dataset.
2. **`notebooks/02_eda.ipynb`** — class distribution, image samples, box stats, t-SNE.
3. **`notebooks/03_train_baseline.ipynb`** (and `updated_03` / `updated_05`) — train
   the baseline and the improved Ghost+MPCA+SIoU model. ~30–60 min+ on GPU.
4. **`notebooks/04_evaluate.ipynb`** — mAP, per-class metrics, confusion matrix,
   sample predictions, **Eigen-CAM explanations**.

### Demo, explainability & deployment
```bash
streamlit run src/app.py                                    # upload/webcam → detect + Eigen-CAM
python src/export_model.py --weights results/baseline_640/weights/best.pt --half --validate
```
Hugging Face Space + edge/mobile export are documented in **`docs/DEPLOYMENT.md`**;
model details are in **`docs/model_card.md`**.

## Results (measured)

Production model: **YOLOv8n baseline @ imgsz 640** (`results/baseline_640/weights/best.pt`).
TEST split (180 imgs, stratified 8:1:1). Chosen over the "improved" architectures after a
5-seed statistical gate — see `experiments/LEADERBOARD.md` and `docs/audit/`.

**Model comparison — fair recipe @640, TEST mAP@0.5:**

| Model | TEST mAP@0.5 | Params | GFLOPs |
|---|---|---|---|
| **YOLOv8n baseline** 🏆 | **0.7525** | 3.01M | 8.09 |
| Ghost+ResCBAM+WIoU (LZY) | 0.7316 | 4.05M | 10.17 |
| Ghost+MPCA+SIoU (paper) | 0.7305 | 2.39M | 6.24 |

5-seed validated: baseline **0.7475 ± 0.0161**; P2-head and YOLOv8s both trend *worse* and
are not significant → **no candidate beats the plain baseline.**

**Deployment benchmark — RTX 2000 Ada, imgsz 640, no TTA** (`docs/audit/DEPLOYMENT_BENCHMARK.md`):

| Backend | Precision | Latency | FPS | TEST mAP@0.5 | Size |
|---|---|---|---|---|---|
| **PyTorch** 🏆 | FP32 | 5.73 ms | 174.6 | **0.7525** | 6.26 MB |
| ONNX | FP32 | 9.96 ms | 100.4 | 0.7041 | 12.27 MB |
| TensorRT | FP16 | 8.06 ms | 124.1 | 0.704 | 9.84 MB |

> **Ship PyTorch FP32** — it is both the fastest *and* most accurate here. TensorRT/ONNX both
> regress to ~0.704 (a shared Ultralytics export-path parity gap, *not* FP16 loss) and TRT is
> ~40% slower because YOLOv8n is too small for graph-opt to pay off on this GPU.

Hardest class is **crazing** (~0.44, low-contrast — the architecture-invariant floor);
easiest are *patches* (0.93) and *pitted_surface* (0.86).

## Reference Papers

- "A lightweight algorithm for steel surface defect detection using improved
  YOLOv8" — *Scientific Reports* (2025), open access.
- "Steel surface defect detection based on improved YOLOv8" — *ResearchGate* (2025).
- "MPA-YOLO: Steel surface defect detection based on improved YOLOv8 framework"
  — *Pattern Recognition* (2025).

## Author

Steel Surface Defect Detection — DEPI AI Track Graduation Project
