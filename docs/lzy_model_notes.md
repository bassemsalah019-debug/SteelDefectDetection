# LZY-233 Improved YOLOv8n — Reproduction Notes

Reproduction of the GitHub project:

> **[LZY-233/yolov8_Imporved-Defect_detection](https://github.com/LZY-233/yolov8_Imporved-Defect_detection)**
> Steel-defect "improved YOLOv8": GhostConv backbone + ResBlock_CBAM head + WIoU loss.
> Reported on NEU-DET: **mAP@0.5 = 0.792**, params **4.05M**, FLOPs **10.2G**.

This is a **second** improved model alongside the *Scientific Reports* paper model
(`docs/improved_model_notes.md`, notebook 05). It is implemented with the same clean,
no-fork pattern: custom modules in `src/modules/`, activated at runtime via
`register_lzy()` — the installed `ultralytics` package is never edited (unlike the
upstream repo, which vendors a modified `ultralytics/`).

---

## 1. The three modifications (from their source)

| # | Modification | Where | How it's implemented here |
|---|--------------|-------|---------------------------|
| 1 | **GhostConv** | the **three stride-2 downsample convs** only (P3/P4/P5); C2f blocks kept standard | `configs/yolov8n_lzy.yaml` (GhostConv is built into Ultralytics). Lighter touch than the paper's full Ghost backbone — which is why this model is *heavier*. |
| 2 | **ResBlock_CBAM** | after **each** of the four neck C2f blocks; Detect reads the 3 CBAM outputs (P3/P4/P5) | `src/modules/rescbam.py` — 1×1→3×3→1×1 bottleneck (BN+LeakyReLU) → CBAM (channel + spatial) → residual add → ReLU. Ported verbatim from their `conv.py`, made channel-preserving + lazy-built. |
| 3 | **WIoU loss** | box regression | `src/modules/wiou.py` — patches `BboxLoss.forward`. Ported from their `IoU_Cal` (`utils/iou.py`): `WIoU = exp(l2_center / l2_box.detach()) · (1−IoU)`, monotonic focusing `·√(iou_loss/iou_mean)` with momentum `1−0.5^(1/7000)`. |

**Architecture (matches their `yolov8_ResBlock_CBAM_GhostConv.yaml`, with `nc=6`):**
backbone Ghost-izes layers 3/5/7; head inserts ResBlock_CBAM at layers 13/17/21/25;
`Detect([17, 21, 25])`. Verified build: **4.057M params**, 4× ResBlock_CBAM — matches their
reported ~4.05M.

---

## 2. Differences vs. the *Scientific Reports* paper model (notebook 05)

| | Paper (nb 05) | LZY-233 (nb 06) |
|---|---|---|
| Lightweight conv | Ghost backbone (Conv→GhostConv **and** C2f→C3Ghost) | GhostConv on 3 downsample convs only |
| Attention | MPCA at end of backbone | ResBlock_CBAM ×4 in the head |
| Loss | SIoU | WIoU |
| Params | ~2.4M (lighter than baseline) | ~4.06M (heavier than baseline) |
| Activation | `register()` | `register_lzy()` |

---

## 3. Training configuration (fair comparison)

Trained with the **same recipe as the baseline** (notebook 03) and the paper model (nb 05) so
any metric difference reflects the **architecture**: `imgsz=640`, `epochs=150`, `batch=16`,
`seed=42`, `patience=50`, `cos_lr=True`, `optimizer='SGD'`, `workers=0` (Windows-safe), COCO transfer from
`yolov8n.pt`, on the paper's **8:1:1** split (1440/180/180). Results → `results/lzy_640/`; the
**test**-set mAP is the number to compare with the repo's reported 0.792.

> **Faithfulness caveat:** the modules are ported from the LZY-233 source (ResBlock_CBAM/CBAM
> verbatim; WIoU from their `IoU_Cal`), but made channel-preserving/lazy and run via runtime
> patching rather than their vendored `ultralytics/` fork. The architecture (4.057M params)
> matches; exact numbers may differ slightly from their run (different split realization,
> their `batch=4`/`amp=False` vs our `batch=16`).

---

## 4. Files added by this reproduction

| Path | Purpose |
|------|---------|
| `src/modules/rescbam.py` | ResBlock_CBAM + CBAM + channel/spatial attention. |
| `src/modules/wiou.py` | WIoU loss + `BboxLoss.forward` patch. |
| `src/modules/__init__.py` | adds `register_lzy()` (wires ResBlock_CBAM + WIoU at runtime). |
| `configs/yolov8n_lzy.yaml` | GhostConv backbone + ResBlock_CBAM head architecture. |
| `notebooks/06_train_lzy.ipynb` | train + val + test, writes `results/lzy_640/metrics_summary.txt`. |
