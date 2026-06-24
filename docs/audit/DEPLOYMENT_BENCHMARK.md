# DEPLOYMENT_BENCHMARK.md

Model: `results\baseline_640\weights\best.pt` | imgsz 640 | device cuda:0 (RTX 2000 Ada) | no TTA. Generated 2026-06-16.

| Backend | Precision | Latency (ms) | FPS | TEST mAP@0.5 | mAP@.5:.95 | Size (MB) |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | 5.73 | 174.6 | 0.7525 | 0.3926 | 6.26 |
| ONNX (ORT 0) | FP32 | 9.96 | 100.4 | 0.7041 | 0.3583 | 12.27 |
| TensorRT | FP16 | 8.06 | 124.1 | 0.704 | 0.3593 | 9.84 |

## Verdict: ship PyTorch FP32

On this hardware/model, **PyTorch FP32 is both the fastest and the most accurate backend** — there is no accuracy/latency trade to make. The TensorRT FP16 engine now builds cleanly (see below) but does not beat it:

- **Latency:** TRT FP16 is 8.06 ms vs PyTorch 5.73 ms — TensorRT is **~40% slower**, not faster. YOLOv8n is tiny (6.26 MB / ~8.7 GFLOPs); kernel-launch + memcpy overhead dominates and TRT's graph optimizations don't pay off at this scale on the Ada. PyTorch's fused CUDA path is already optimal here.
- **Accuracy:** TRT FP16 lands at **0.704 mAP@0.5 (−4.85 pp vs PyTorch 0.7525)** — and ONNX FP32 lands at almost exactly the same number (0.7041). Because both exported backends regress to the *same* value while only PyTorch's native `.val()` gives 0.7525, this is **not an FP16 precision loss** — it's a systematic delta in the Ultralytics export eval path (letterbox padding / NMS / output-decode), affecting ONNX and TRT identically. FP16 quantization is essentially lossless here (TRT 0.7040 vs ONNX FP32 0.7041).

## TensorRT build resolution
- TRT **11.0** export fails: Ultralytics references `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`, which TRT 11 removed (API breakage, not a model issue).
- Fix: downgrade to **`tensorrt==10.7.0`** (with `ultralytics 8.4.51`). `EXPLICIT_BATCH` restored → FP16 engine builds in 228 s, 9.4 MB. Verified end-to-end (build + val + latency).

## Notes / next steps
- **Production backend = PyTorch FP32** (`results/baseline_640/weights/best.pt`, 0.7525 mAP, 174 FPS). No export needed for real-time use.
- The exported-eval −4.85 pp regression (shared by ONNX + TRT) is a **preprocessing/decode parity gap in the export path**, not quantization. If a non-PyTorch backend is ever required (CPU demo, edge), fix the letterbox/NMS delta first — don't trust the exported mAP until then.
- The `.engine` file (`results/baseline_640/weights/best.engine`) is retained for reference but is **not** the recommended serving artifact on this hardware.
