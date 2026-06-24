"""
export_model.py - Export a trained model to a deployment format and validate parity.

The recommended path is ONNX (FP16) as a hub format: from ONNX you can target
ONNX Runtime (desktop / server / Android), NCNN (Raspberry Pi / ARM), and TFLite
(Android). TensorRT is only needed for NVIDIA Jetson.

Examples
--------
    # Export the best model to ONNX and confirm accuracy is preserved:
    python src/export_model.py --weights results/improved_opt/weights/best.pt --validate

    # Half-precision ONNX at 640, no validation:
    python src/export_model.py --weights results/improved_opt/weights/best.pt --half

    # Raspberry Pi target:
    python src/export_model.py --weights results/improved_opt/weights/best.pt --format ncnn

The custom MPCA layer is made of standard conv/pool ops and exports cleanly, but
--validate is strongly recommended before shipping: it re-evaluates the exported
model on the held-out TEST split and prints the PyTorch-vs-export mAP@0.5 delta.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.explain import enable_custom_modules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained YOLO model and validate parity.")
    ap.add_argument("--weights", required=True, help="path to best.pt (relative to project root or absolute)")
    ap.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "tflite", "ncnn", "engine"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 export (smaller / faster)")
    ap.add_argument("--validate", action="store_true", help="compare PyTorch vs export on the TEST split")
    ap.add_argument("--data", default="data/neu-det-yolo/data.yaml")
    args = ap.parse_args()

    from ultralytics import YOLO

    enable_custom_modules()
    weights = ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    assert weights.exists(), f"weights not found: {weights}"

    print(f"Exporting {weights.name} -> {args.format}  (imgsz={args.imgsz}, half={args.half})")
    out = YOLO(str(weights)).export(format=args.format, imgsz=args.imgsz, half=args.half, simplify=True)
    print("Exported:", out)

    if args.validate:
        data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
        assert data.exists(), f"data config not found: {data}"
        print("\nValidating PyTorch vs export on the TEST split...")
        pt = YOLO(str(weights)).val(data=str(data), split="test", imgsz=args.imgsz, verbose=False)
        ex = YOLO(str(out)).val(data=str(data), split="test", imgsz=args.imgsz, verbose=False)
        d = float(ex.box.map50) - float(pt.box.map50)
        print(f"  PyTorch    mAP@0.5 = {float(pt.box.map50):.4f}")
        print(f"  {args.format:<9} mAP@0.5 = {float(ex.box.map50):.4f}")
        print(f"  delta              = {d:+.4f}  {'OK' if abs(d) < 0.01 else 'CHECK EXPORT'}")


if __name__ == "__main__":
    main()
