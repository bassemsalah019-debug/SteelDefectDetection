---
title: Steel Surface Defect Detection
emoji: 🔩
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Steel Surface Defect Detection Studio (YOLOv8n + Adaptive thresholding + Eigen-CAM + bilingual report)

Automated detection of six surface-defect types on hot-rolled steel strips
(NEU-DET), in a dark "Molten Graphite" **Detection Studio** UI. Upload an image or use
your webcam; the app collapses it to grayscale (matching how the model was trained),
draws bounding boxes, lists the defects with confidence, shows an **Eigen-CAM** heatmap
of where the model looked, and generates a grounded **bilingual (English / Arabic)
defect report** you can download as HTML or PDF.

Two inference modes: **Fixed** (one global confidence) and **Adaptive** — a per-class,
per-image confidence threshold from class difficulty, image brightness, image quality
and detection density (post-processing only, so it works on PyTorch *and* exported
ONNX/TensorRT). See `docs/audit/ADAPTIVE_THRESHOLDING.md`.

**Classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

**Model:** plain **YOLOv8n baseline** — the production choice after a controlled,
5-seed study: TEST mAP@0.5 **0.7525** (5-seed mean 0.7475 ± 0.016), 6.3 MB, ~157 FPS on
an RTX 2000 Ada. A reproduction study found lightweight "improvements" (Ghost/MPCA/SIoU,
CBAM/WIoU) did **not** beat this baseline on NEU-DET — recipe, not architecture, drove the gains.

> This is the **public CPU demo**. Detection + Eigen-CAM run on CPU; the report uses the
> grounded knowledge base (no live LLM on the free tier — it degrades gracefully). The full
> on-prem stack (TensorRT FP16 + local MiMo-7B for live reports) runs on the GPU host.
>
> This `README.md` is also the Space card (the YAML header configures the Space). To
> (re)deploy: `python deployment/huggingface/build_space.py` then push `space/`. Full
> steps in `docs/DEPLOYMENT.md`.
