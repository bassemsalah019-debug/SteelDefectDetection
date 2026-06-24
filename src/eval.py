"""
eval.py - the one measurement harness for the leaderboard.

Evaluates a trained checkpoint on the held-out TEST split and writes a single
`results.json` with every metric the master prompt requires:
  mAP@0.5, mAP@0.5:0.95, precision, recall, per-class AP@0.5, params, GFLOPs,
  FPS (with the device stated), and model size (MB).

This is a GPU job when run with device=0 - per the project rule, YOU run it in
the notebook / shell; this file only prepares it. It imports cleanly without a GPU.

    python -m src.eval --weights results/improved_opt/weights/best.pt \
        --data configs/neu_det.yaml --split test --imgsz 800 --device 0

Then the row is ready to paste into experiments/LEADERBOARD.md (every number traced
to the run's results.json).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _register_custom_modules() -> None:
    """Make improved / LZY checkpoints un-pickle. Harmless for stock models."""
    try:
        from src.modules import register, register_lzy

        register(verbose=False)
        register_lzy(verbose=False)
    except Exception:
        pass


def _per_class_ap50(metrics, names: dict[int, str]) -> dict[str, float]:
    """Per-class AP@0.5 keyed by class name, robust across ultralytics versions."""
    box = metrics.box
    try:
        idx = list(getattr(box, "ap_class_index", []))
        ap50 = getattr(box, "ap50", None)
        if ap50 is not None and len(idx) == len(ap50):
            return {names.get(int(c), str(c)): round(float(a), 4) for c, a in zip(idx, ap50)}
    except Exception:
        pass
    # fallback: mAP@0.5:0.95 per class (box.maps) aligned to names
    try:
        return {names.get(i, str(i)): round(float(m), 4) for i, m in enumerate(box.maps)}
    except Exception:
        return {}


def _measure_fps(model, imgsz: int, device, n: int = 60, warmup: int = 10) -> float:
    """Frames/sec on `device` using a representative grayscale-replicated input."""
    import numpy as np

    dummy = np.full((imgsz, imgsz, 3), 127, dtype="uint8")  # neutral gray, 3ch
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    dt = time.perf_counter() - t0
    return n / dt if dt > 0 else float("nan")


def _device_name(device) -> str:
    import torch

    if str(device) not in ("cpu", "-1", "None") and torch.cuda.is_available():
        try:
            return f"cuda:{device} ({torch.cuda.get_device_name(int(device))})"
        except Exception:
            return f"cuda:{device}"
    return "cpu"


def evaluate(weights: str, *, data: str = "configs/neu_det.yaml", split: str = "test",
             imgsz: int = 800, device: str | int = 0, batch: int = 16,
             measure_fps: bool = True, out: str | None = None) -> dict:
    """Run validation + cost measurement and write results.json. Returns the dict."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    _register_custom_modules()
    wpath = (ROOT / weights) if not Path(weights).is_absolute() else Path(weights)
    dpath = (ROOT / data) if not Path(data).is_absolute() else Path(data)
    model = YOLO(str(wpath))

    metrics = model.val(data=str(dpath), split=split, imgsz=imgsz, device=device,
                        batch=batch, verbose=False, plots=False)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    params = int(get_num_params(model.model))
    try:
        gflops = round(float(get_flops(model.model, imgsz)), 2)
    except Exception:
        gflops = None
    fps = round(_measure_fps(model, imgsz, device), 1) if measure_fps else None

    result = {
        "weights": str(wpath.relative_to(ROOT)) if wpath.is_relative_to(ROOT) else str(wpath),
        "data": str(dpath), "split": split, "imgsz": imgsz,
        "device": _device_name(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class_AP50": _per_class_ap50(metrics, names),
        "params": params,
        "params_millions": round(params / 1e6, 3),
        "gflops": gflops,
        "model_size_mb": round(wpath.stat().st_size / 1e6, 2) if wpath.exists() else None,
        "fps": fps,
        "latency_ms": round(1000.0 / fps, 2) if fps else None,
        "fps_device": _device_name(device),
    }

    out_path = Path(out) if out else wpath.parent.parent / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["_results_json"] = str(out_path)
    return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint -> results.json")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/neu_det.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-fps", action="store_true", help="skip FPS timing")
    ap.add_argument("--out", default=None, help="results.json path (default: run folder)")
    args = ap.parse_args()

    res = evaluate(args.weights, data=args.data, split=args.split, imgsz=args.imgsz,
                   device=args.device, batch=args.batch, measure_fps=not args.no_fps,
                   out=args.out)
    m = res["metrics"]
    print(f"\n{res['weights']}  [{res['split']}] @ imgsz {res['imgsz']} on {res['device']}")
    print(f"  mAP@0.5 {m['mAP50']:.4f} | mAP@0.5:0.95 {m['mAP50_95']:.4f} "
          f"| P {m['precision']:.4f} | R {m['recall']:.4f}")
    print(f"  params {res['params_millions']}M | GFLOPs {res['gflops']} "
          f"| {res['fps']} FPS | {res['model_size_mb']} MB")
    print(f"  -> {res['_results_json']}")


if __name__ == "__main__":
    _cli()
