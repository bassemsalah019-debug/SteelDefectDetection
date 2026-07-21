# Steel Surface Defect Detection

Automated detection of **6 surface-defect types** on hot-rolled steel strips (NEU-DET) with
**YOLOv8n** — from research (a rigorous, 5-seed-validated study) to two working, deployed
demos. DEPI AI Track graduation capstone.

**Defect classes:** crazing · inclusion · patches · pitted_surface · rolled-in_scale · scratches

---

## 🚩 Flagship: SteelVision — full-stack inspection platform

A login-protected web app (**FastAPI + React/TypeScript**) where an inspector uploads steel
images (single or batch), the system detects defects (**Fixed or Adaptive** thresholds),
shows **Eigen-CAM** heatmaps, generates a grounded **bilingual (EN/AR)** report, saves every
inspection, and charts defect trends on a dashboard.

- **Code:** [`webapp/`](webapp/) — [backend](webapp/backend/) · [frontend](webapp/frontend/) · [run & test guide](webapp/README.md)
- **Live demo:** https://huggingface.co/spaces/hazemaaa/steelvision
- **Run locally:** double-click `webapp/start-app.bat` → http://localhost:5173 · **test login** `inspector@steel.io` / `password123`
- Tests (15 passing), Alembic migrations, Docker Compose (Postgres) all included.

## 🔬 The research

- **Model:** YOLOv8n baseline — **TEST mAP@0.5 0.7525** (5-seed **0.7475 ± 0.016**), 6 MB, ~157 FPS → [`models/`](models/)
- **Honest finding:** popular "improved YOLOv8" variants (Ghost+MPCA+SIoU, CBAM+WIoU, P2-head, YOLOv8s) **did not beat the plain baseline** under a fair, recipe-matched, 5-seed statistical gate.
- **Adaptive Real-Time Confidence Thresholding:** per-class, per-image thresholds from class difficulty + brightness + quality + detection density (post-processing only; ONNX/TRT-safe) → [`docs/audit/ADAPTIVE_THRESHOLDING.md`](docs/audit/ADAPTIVE_THRESHOLDING.md)
- **Notebooks:** [`notebooks/`](notebooks/) — data prep, EDA, training, evaluation, 5-seed study
- **Full audit trail:** [`docs/audit/`](docs/audit/) · **model card:** [`docs/model_card.md`](docs/model_card.md)

## 🎛️ Secondary demo: Streamlit Studio

Single-page research/XAI demo (detection + adaptive thresholding + Eigen-CAM + bilingual report).
- **Code:** [`src/app.py`](src/app.py) · **Live:** https://huggingface.co/spaces/hazemaaa/steel-defect-detection
- **Run:** `streamlit run src/app.py`

---

## Repository layout

```
SteelDefectDetection/
├── models/            # ⭐ the production model (best.pt) + model card
├── webapp/            # 🚩 SteelVision — FastAPI backend + React frontend (flagship)
├── src/               # shared ML: infer · adaptive_threshold · explain · report · modules · app.py (Streamlit)
├── notebooks/         # data prep, EDA, training, evaluation, 5-seed study (8 notebooks)
├── configs/           # dataset + model-architecture YAMLs
├── scripts/           # eval / benchmark / audit / seed-study scripts
├── tests/             # preprocessing-parity, adaptive-threshold, smoke tests
├── experiments/       # 5-seed study METRICS (json/csv/yaml)  — leaderboard evidence
├── deployment/        # Hugging Face Space build (Streamlit)
└── docs/              # model card · audit/ · deployment · presentation
```

## Dataset (not in git — regenerable)

**NEU-DET** — 1,800 grayscale 200×200 images, 6 classes × 300, stratified **8:1:1** split.
Auto-downloaded via `kagglehub` in [`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb)
(or Kaggle: *kaustubhdikshit/neu-surface-defect-database*). `data/`, `results/`, and training
weights are git-ignored (large/regenerable); the deployed model is kept in [`models/`](models/).

## Setup

```bash
pip install -r requirements.txt          # research/ML deps (torch installed separately)
# SteelVision app: see webapp/README.md
```

## Results (measured, TEST split)

| Model | TEST mAP@0.5 | Params | GFLOPs |
|---|---|---|---|
| **YOLOv8n baseline** 🏆 | **0.7525** (5-seed 0.7475 ± 0.016) | 3.01M | 8.09 |
| Ghost+ResCBAM+WIoU (LZY) | 0.7316 | 4.05M | 10.17 |
| Ghost+MPCA+SIoU (paper) | 0.7305 | 2.39M | 6.24 |

Hardest class: **crazing** (~0.44, low-contrast floor). Easiest: patches (0.93), pitted_surface (0.86).
Deployment benchmark and full analysis in [`docs/audit/`](docs/audit/).
