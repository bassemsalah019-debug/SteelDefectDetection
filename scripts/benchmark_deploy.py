"""
benchmark_deploy.py - deployment export + benchmark for the shippable model.

Exports the chosen checkpoint to ONNX and (if TensorRT is installed) a TensorRT
FP16 engine, then benchmarks every available backend on the RTX 2000 Ada:
latency (ms), FPS, TEST mAP@0.5 retained (no TTA), and file size (MB). Each
backend is independent and skipped gracefully if its toolchain is absent.

GPU note: PyTorch + TensorRT benchmarks run on cuda:0. This is an inference/export
job (not training); it's cheap (seconds-minutes). Writes
docs/audit/DEPLOYMENT_BENCHMARK.md + .json.

    python scripts/benchmark_deploy.py --weights results/baseline_opt/weights/best.pt --imgsz 640
    python scripts/benchmark_deploy.py ... --engine     # also build TensorRT FP16 (needs `tensorrt`)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "docs" / "audit" / "DEPLOYMENT_BENCHMARK.md"
OUT_JSON = ROOT / "docs" / "audit" / "deployment_benchmark.json"


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _register():
    try:
        from src.modules import register, register_lzy
        register(verbose=False); register_lzy(verbose=False)
    except Exception:
        pass


def measure_latency(model, imgsz: int, device, n: int = 100, warmup: int = 10) -> dict:
    import numpy as np
    dummy = np.full((imgsz, imgsz, 3), 127, dtype="uint8")
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    dt = (time.perf_counter() - t0) / n
    return {"latency_ms": round(dt * 1000, 2), "fps": round(1.0 / dt, 1)}


def eval_map(model, data: Path, imgsz: int, device) -> dict:
    m = model.val(data=str(data), split="test", imgsz=imgsz, device=device,
                  verbose=False, plots=False)
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}


def bench_backend(name, path, imgsz, data, device, precision):
    from ultralytics import YOLO
    _register()
    model = YOLO(str(path))
    row = {"backend": name, "precision": precision,
           "size_MB": round(Path(path).stat().st_size / 1e6, 2)}
    try:
        row.update(measure_latency(model, imgsz, device))
    except Exception as e:
        row["latency_ms"] = None; row["fps"] = None; row["latency_err"] = str(e)[:120]
    try:
        row.update(eval_map(model, data, imgsz, device))
    except Exception as e:
        row["mAP50"] = None; row["map_err"] = str(e)[:120]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/baseline_opt/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default="configs/neu_det.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--engine", action="store_true", help="also build+bench TensorRT FP16 (needs tensorrt)")
    ap.add_argument("--onnx-only", action="store_true", help="skip GPU; ONNX/CPU portable path only")
    args = ap.parse_args()

    import sys; sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO
    _register()

    weights = ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    assert weights.exists(), f"weights not found: {weights}"
    rows, notes = [], []

    # 1) PyTorch (FP32) on GPU - always available, the reference
    if not args.onnx_only:
        print("== PyTorch FP32 @ cuda ==")
        rows.append(bench_backend("PyTorch", weights, args.imgsz, data, args.device, "FP32"))

    # 2) ONNX (portable reference) - needs `onnx`; runs on ORT (CPU here unless onnxruntime-gpu)
    if have("onnx"):
        print("== exporting ONNX ==")
        try:
            onnx_path = YOLO(str(weights)).export(format="onnx", imgsz=args.imgsz, simplify=True)
            ort_dev = "cpu"  # CPU ORT installed; set to 0 if onnxruntime-gpu present
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    ort_dev = args.device
            except Exception:
                pass
            rows.append(bench_backend(f"ONNX (ORT {ort_dev})", onnx_path, args.imgsz, data, ort_dev, "FP32"))
        except Exception as e:
            notes.append(f"ONNX export failed: {e}")
    else:
        notes.append("ONNX skipped: `pip install onnx onnxslim` to enable the portable reference export.")

    # 3) TensorRT FP16 engine - needs `tensorrt`; build ON this GPU
    if args.engine:
        if have("tensorrt"):
            print("== building TensorRT FP16 engine ==")
            try:
                eng = YOLO(str(weights)).export(format="engine", imgsz=args.imgsz, half=True, device=args.device)
                rows.append(bench_backend("TensorRT", eng, args.imgsz, data, args.device, "FP16"))
            except Exception as e:
                notes.append(f"TensorRT build failed: {e}")
        else:
            notes.append("TensorRT skipped: `pip install tensorrt` (~1 GB) to build the FP16 engine.")

    # ---- write table ----
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    hdr = "| Backend | Precision | Latency (ms) | FPS | TEST mAP@0.5 | mAP@.5:.95 | Size (MB) |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [f"# DEPLOYMENT_BENCHMARK.md", f"",
             f"Model: `{weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}` | imgsz {args.imgsz} | "
             f"device cuda:{args.device} (RTX 2000 Ada) | no TTA. Generated {time.strftime('%Y-%m-%d')}.", "",
             hdr, sep]
    for r in rows:
        lines.append(f"| {r['backend']} | {r['precision']} | {r.get('latency_ms')} | {r.get('fps')} | "
                     f"{r.get('mAP50')} | {r.get('mAP50_95')} | {r.get('size_MB')} |")
    if notes:
        lines += ["", "## Notes / next steps"] + [f"- {n}" for n in notes]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"rows": rows, "notes": notes}, indent=2), encoding="utf-8")

    print("\n" + "\n".join(lines[3:]))
    print(f"\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
