# DEPLOYMENT_RECOMMENDATIONS.md
*Generated 2026-06-13. [M]=measured today on the RTX 2000 Ada; [HYP]=needs the export/benchmark run.*

## 1. Measured baseline cost (PyTorch, imgsz 640, cuda:0)
[M] From the eval harness `Speed:` line:
| Model | preprocess | inference | postprocess | ≈total | ≈FPS (infer) | GFLOPs | size |
|---|---|---|---|---|---|---|---|
| YOLOv8n baseline | 1.3 ms | **6.7 ms** | 2.5 ms | ~10.5 ms | ~149 | 8.09 | 6.3 MB |
| paper (Ghost+MPCA+SIoU) | 1.2 | 6.6 | 2.7 | ~10.5 | ~150 | 6.24 | 5.1 MB |
| LZY | 1.2 | 7.8 | 2.4 | ~11.4 | ~128 | 10.17 | 8.4 MB |

**Implication [M]:** all three already exceed real-time (>100 FPS) on the Ada in plain PyTorch.
The Ghost model's lower GFLOPs does **not** translate into a meaningful latency win here (6.6 vs
6.7 ms). **On this hardware, lightweighting buys nothing** — it only matters if the target later
changes to a Jetson/edge-MCU.

## 2. Recommended deployment path (target = RTX 2000 Ada / CUDA)
1. **ONNX** (portable reference) — already supported by `src/export_model.py`. Use opset 12+, fixed
   imgsz, dynamic batch optional.
2. **TensorRT FP16** — the production engine. [HYP] expect ~2–3× throughput over PyTorch FP32 and
   negligible mAP loss (FP16 is safe). **Build on the Ada itself** (engines are HW/driver-specific).
3. **TensorRT INT8** — only if multi-stream throughput is needed. [HYP] **RISK: accuracy** — the weak
   low-contrast classes (`crazing`, `rolled-in_scale`) are the most likely to degrade under INT8.
   Calibrate with a representative subset and **re-score on TEST**; adopt only if mAP@0.5 drop < ~0.5 pp.
4. **Do NOT** ship TFLite/NCNN/OpenVINO as primary — wrong target. Keep as documented one-config-away
   options for a future ARM/edge port.

## 3. Public demo vs on-prem (two surfaces)
- **On-prem (real industrial):** TensorRT FP16 engine + local MiMo-7B for reports, both on the Ada.
- **Public demo (HF Spaces free / no GPU):** detection + Eigen-CAM on **CPU via ONNX**; the LLM report
  must point at a hosted endpoint (`STEEL_LLM_BASE_URL`) or gracefully degrade — which `src/report/`
  already does. Don't advertise the GPU/local-LLM path on the free demo.

## 4. Gate-required steps (not yet done)
| Step | Gate | Owner |
|---|---|---|
| Install TensorRT toolchain | download | you (approve) |
| Build FP16 + INT8 engines | GPU | you |
| INT8 calibration subset | (exists in data plan) | data-engineer |
| Benchmark table: FP32/FP16/INT8 × latency/FPS/mAP-retained/size | GPU | you run, I tabulate |

## 5. Which checkpoint to deploy
[M] **Deploy the YOLOv8n baseline** (`baseline_opt` for the @800 accuracy, or `baseline_640` for
lowest latency) — it is the measured accuracy leader at the fair recipe and the lightest *effective*
choice on this GPU. Re-confirm the choice after the 5-seed study.
