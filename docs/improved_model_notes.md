# Improved YOLOv8n — Paper Reproduction Notes

Reproduction of the architecture from:

> **A lightweight algorithm for steel surface defect detection using improved YOLOv8**
> *Scientific Reports* (2025) — [s41598-025-93469-5](https://www.nature.com/articles/s41598-025-93469-5)

The goal is a **faithful, architecture-only** reproduction to serve as a clean reference
point against the existing stock YOLOv8n baseline (`results/baseline`). No extra tricks
were added beyond what the paper specifies.

---

## 1. Exact paper modifications implemented

The paper makes **three** changes to baseline YOLOv8n. All three are reproduced:

| # | Modification | Where (paper) | How it's implemented here |
|---|--------------|---------------|---------------------------|
| 1 | **Ghost backbone** — backbone `Conv → GhostConv` and `C2f → C3Ghost` (GhostNet, Han et al. 2020) | Backbone only; neck/head unchanged | `configs/yolov8n_improved.yaml`. `GhostConv`/`C3Ghost` are **built into Ultralytics** — no custom code. |
| 2 | **MPCA attention** — MultiPath Coordinate Attention | **End of the backbone** (after SPPF) | Custom module `src/modules/mpca.py`, inserted as backbone layer `10`. |
| 3 | **SIoU loss** — replaces CIoU box regression | Box regression branch | Custom patch `src/modules/siou.py`; monkey-patches `BboxLoss.forward`. |

**Mapping to the YAML** (`configs/yolov8n_improved.yaml`):

```
backbone:
  0  Conv       (stem, kept standard)
  1  GhostConv  3-P2/4            #  modification 1
  2  C3Ghost                      #  modification 1
  3  GhostConv  3-P3/8
  4  C3Ghost              <- P3   (to neck)
  5  GhostConv  5-P4/16
  6  C3Ghost              <- P4   (to neck)
  7  GhostConv  7-P5/32
  8  C3Ghost
  9  SPPF
  10 MPCA                         #  modification 2  (end of backbone)
head: standard YOLOv8 PAN-FPN (Conv + C2f), indices shifted +1; deepest path
      starts from the MPCA output (layer 10). Detect on layers [16, 19, 22].
```

The paper's reported effect: params **3.01M → 2.04M (−32%)**, GFLOPs **8.1 → 5.1 (−37%)**,
mAP@0.5 **0.774 → 0.786 (+1.2 pp)**. This reproduction lands at **~2.40M params** (−20% vs
the 3.01M baseline) — lighter than baseline and in the same ballpark as the paper (the small
gap is expected: we Ghost-ize the **backbone only**, per the paper text, and keep the
standard C2f neck).

---

## 2. What the paper did **not** specify (defaults used)

These were not stated in the paper; the choice made here is documented so paper-derived
details are distinguishable from assumptions.

| Item | `[NOT SPECIFIED]` → default used | Rationale |
|------|----------------------------------|-----------|
| MPCA internal dims | reduction ratio `r = 32` | Coordinate-Attention (Hou 2021) default. |
| MPCA path count | `4` paths | The paper explicitly says "four paths". |
| MPCA path fusion | sum of paths → 1×1 conv (**no input residual**) | Paper combines paths by "addition"; the 1×1 conv is the "MLP" combination. Kept paper-literal — no extra residual added. |
| MPCA pooling op | `mean()` (not `AdaptiveAvgPool2d`) | Mathematically identical, but has a deterministic CUDA backward so it runs cleanly under Ultralytics' seeded/deterministic mode (no warning spam). |
| MPCA exact layer | **after SPPF** (backbone layer 10) | Paper says "at the end of the backbone network". |
| Stem conv (layer 0) | kept as **`Conv`**, not GhostConv | Matches Ultralytics' official `yolov8-ghost.yaml`; GhostConv on a 3-channel input is degenerate. |
| SIoU shape exponent `θ` | `θ = 4` | Gevorgyan (2022) recommended default (range 2–6); the paper does not print θ. |
| Dynamic NMS threshold | **not applied** (default val NMS used) | The paper mentions a dynamic NMS of 0.6 at inference, but this is post-processing, not architecture. Leaving val NMS at the baseline default keeps the comparison architecture-only and fair. |
| epochs / batch / imgsz / optimizer / lr | taken from **this project's baseline** | The paper does not give them; matching the baseline run is required for a fair comparison (see §3). |

> **Faithfulness caveat on MPCA:** the paper describes MPCA only conceptually (four paths,
> horizontal + vertical pooling, MLP → channel weights, fuse). No layer table or exact
> channel/kernel sizes are given. `src/modules/mpca.py` is a faithful, working
> reconstruction of that description (multi-path Coordinate Attention); it is **not** a
> bit-exact copy of the authors' (unreleased) code.

---

## 3. Training configuration (fair comparison)

The improved model is trained with the **same settings as the baseline** (notebook 03,
`results/baseline_640/`) on the paper's **8:1:1** split, so any metric difference reflects the
**architecture**, not the recipe:

| setting | value |
|---------|-------|
| imgsz | **640** (paper resolution; NEU-DET 200×200 upscaled) |
| epochs | **150** |
| batch | **16** (fits 16 GB VRAM at 640) |
| seed | 42 |
| patience | **50** (protects the late `close_mosaic` boost from a noisy mid-run dip) |
| cos_lr | **True** (cosine LR decay — smoother convergence for the from-scratch backbone) |
| optimizer | **SGD** (lr0=0.01, momentum 0.937, weight_decay 5e-4; standard detection optimizer — `auto` picks AdamW on small data and undershoots final mAP) |
| device | 0 (GPU) |
| workers | **0** (Windows-safe; see note) |
| pretrained | COCO `yolov8n.pt`, transferred where layers are compatible |

> **Windows hang fix (`workers=0`):** with the default `workers=8`, Ultralytics' data-loader
> rebuild at the `close_mosaic` epoch (`epochs − close_mosaic`, i.e. epoch 140 at 150 epochs) can
> **deadlock** on Windows due to multiprocessing worker re-spawn. Setting `workers=0` (single-process
> loading) removes the deadlock; with the dataset cached in RAM it is just as fast, and changes only
> data-loading parallelism — **not the weights or metrics** — so the comparison stays fair.

> **Recipe history:** earlier runs used `imgsz=224` (undershot the paper by ~13 pp) and `epochs=100`
> with one baseline accidentally cut to 55 epochs. The current 640 / 150-epoch / cos_lr recipe is
> applied uniformly to all three models for a fair, fully-trained comparison.

Because the backbone is Ghost-ized, only the **standard neck/head** weights transfer from
`yolov8n.pt`; the Ghost backbone and MPCA train from scratch (Ultralytics prints
`Transferred X/Y items`). This is the expected transfer-learning behaviour for a modified
backbone.

---

## 4. How to run

From the project root, using the existing venv
(`C:\Users\student\Downloads\files\.venv`) — **no new installs, no torch reinstall**:

1. **Train the improved model** — open and run `notebooks/05_train_improved.ipynb`.
   It registers the custom modules, builds from `configs/yolov8n_improved.yaml`, transfers
   COCO weights, trains (~10 min on the 16 GB GPU), and writes the weights to
   `results/improved/weights/best.pt` and the metrics to `results/improved/metrics_summary.txt`.

### Using the custom modules in your own code

```python
import sys; sys.path.insert(0, r"C:\Users\student\Desktop\SteelDefectDetection")
from src.modules import register
register()                                   # activates MPCA + SIoU
from ultralytics import YOLO
model = YOLO("configs/yolov8n_improved.yaml")
model.load("yolov8n.pt")                     # transfer compatible COCO weights
```

`register()` must be called **before** building/loading any improved model or reloading a
saved `best.pt` (so the `MPCA` layer name resolves and the saved checkpoint unpickles).

---

## 5. Files added by this reproduction

| Path | Purpose |
|------|---------|
| `src/modules/__init__.py` | `register()` — wires MPCA + SIoU into Ultralytics at runtime. |
| `src/modules/mpca.py` | MPCA (MultiPath Coordinate Attention) module — paper modification #2. |
| `src/modules/siou.py` | SIoU loss + `BboxLoss.forward` patch — paper modification #3. |
| `configs/yolov8n_improved.yaml` | Improved architecture (Ghost backbone + MPCA, standard neck). |
| `notebooks/05_train_improved.ipynb` | Train improved model, write `results/improved/metrics_summary.txt`. |

Nothing in the installed `ultralytics` package is modified; all changes are applied at
runtime from `src/modules/`.
