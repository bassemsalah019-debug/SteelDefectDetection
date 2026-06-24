# Deployment Guide

How to run, export, and deploy the steel-defect detector. The deployed model is
`results/improved_opt/weights/best.pt` (improved YOLOv8n, Ghost+MPCA+SIoU, test
mAP@0.5 = 0.768). All commands run from the project root with the project venv active.

---

## A) Local Streamlit app

```powershell
.\.venv\Scripts\Activate.ps1          # the venv that has ultralytics
streamlit run src/app.py
```

Features: upload **or** webcam input, model picker, confidence + inference-size
sliders, a detections table, and an Eigen-CAM explainability overlay. The app calls
`enable_custom_modules()` before loading, so the custom Ghost+MPCA+SIoU checkpoint
un-pickles correctly (this was the bug in the original demo).

---

## B) Hugging Face Spaces (free public URL)

The Space files live in `deployment/huggingface/`. The CLI is now `hf` (the old
`huggingface-cli` is deprecated).

1. **Stage the folder** (bundles app + src + weights into `space/`, git-ignored):
   ```powershell
   python deployment/huggingface/build_space.py
   ```
2. **Log in** with a WRITE token from https://huggingface.co/settings/tokens :
   ```powershell
   hf auth login
   ```
3. **Create the Space** at https://huggingface.co/new-space - pick **SDK = Streamlit**.
4. **Upload** the staged folder (one command; handles large files automatically):
   ```powershell
   hf upload <username>/<space-name> deployment/huggingface/space . --repo-type space
   ```
5. Open the Space page; it builds on CPU and serves at a public URL.

`best.pt` is ~5 MB, well under the limit. The CPU `requirements.txt` keeps the image
small. To refresh after retraining: re-run `build_space.py`, then repeat step 4.
Streamlit Community Cloud works similarly (point it at a GitHub repo).

---

## C) Export for edge / mobile (ONNX hub)

Export once to ONNX, then fan out. **Always `--validate`** before shipping so the
custom MPCA layer is confirmed to have exported without accuracy loss.

```powershell
# ONNX (FP16) + parity check on the TEST split:
python src/export_model.py --weights results/improved_opt/weights/best.pt --half --validate

# Raspberry Pi / ARM CPU:
python src/export_model.py --weights results/improved_opt/weights/best.pt --format ncnn

# Android (TFLite):
python src/export_model.py --weights results/improved_opt/weights/best.pt --format tflite

# NVIDIA Jetson (run ON the Jetson):
python src/export_model.py --weights results/improved_opt/weights/best.pt --format engine --half
```

| Target | Format | Why |
|---|---|---|
| Desktop / server / Android | **ONNX Runtime** | one export, cross-platform |
| Raspberry Pi / ARM CPU | **NCNN** | best ARM-CPU FPS |
| Android native / INT8 | **TFLite** | Android-native |
| iPhone / iPad | **CoreML** (`--format coreml`) | native iOS |
| NVIDIA Jetson | **TensorRT** (`engine`) | max GPU FPS |

---

## Estimated FPS (improved model, 6.3 GFLOPs @ 640) - confirm with `model.benchmark()`

| Device | Runtime | Est. FPS | Real-time? |
|---|---|---|---|
| Laptop GPU (RTX 2000 Ada) | PyTorch/ONNX-GPU | 150-250+ | yes |
| Laptop CPU (modern x86) | ONNX Runtime FP16 | 15-35 | yes |
| Android (Snapdragon 7/8) | NCNN / ORT-Mobile | 15-30 | yes |
| iPhone (A14+) | CoreML | 30-60 | yes |
| Raspberry Pi 4 / 5 | NCNN INT8 | 3-8 / 8-15 | borderline / yes |
| Jetson Nano / Orin Nano | TensorRT FP16 | 15-25 / 80-150 | yes |

The model runs on every target listed; only Raspberry Pi 4 is marginal for real-time.
