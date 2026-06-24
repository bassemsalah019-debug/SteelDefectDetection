"""
run_seed_study.py - the statistical-validity protocol (master mandate Phase 9).

WHY: every leaderboard gap on this dataset is currently single-seed on a 180-image
test set. The @800 "improved beats baseline" gap is +0.48 pp - almost certainly inside
the noise. This script makes claims defensible: train the SAME config across 5 seeds,
score each on TEST, and report mean / std / 95% CI. Run it for TWO configs (e.g. the
baseline and a candidate) to get a proper two-sample comparison.

GPU NOTE: `--train` launches real training (your RTX 2000 Ada, one run at a time). Per
project rule you run this; it is NOT auto-launched by the assistant. `--aggregate`
is CPU-only and reads whatever runs exist.

Examples
--------
# 1) Train the 5-seed baseline @640 (the current fair-recipe leader):
python scripts/run_seed_study.py --train --tag baseline_640 --model yolov8n.pt --imgsz 640 --epochs 150
# 2) Train a candidate the same way (e.g. P2 head, or yolov8s):
python scripts/run_seed_study.py --train --tag yolov8s_640 --model yolov8s.pt --imgsz 640 --epochs 150
# 3) Aggregate + compare:
python scripts/run_seed_study.py --aggregate --tag baseline_640
python scripts/run_seed_study.py --aggregate --tag baseline_640 --compare yolov8s_640
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 123, 777, 2025, 3407]
STUDY = ROOT / "experiments" / "seed_study"
DATA = ROOT / "configs" / "neu_det.yaml"
# t critical (two-sided 95%) for small n; index by dof
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 9: 2.262}


def _register():
    try:
        from src.modules import register, register_lzy
        register(verbose=False); register_lzy(verbose=False)
    except Exception:
        pass


def train_one(tag: str, model: str, imgsz: int, epochs: int, seed: int,
              batch: int = 16, cache: bool = True) -> Path:
    import sys
    sys.path.insert(0, str(ROOT))

    name = f"{tag}_seed{seed}"
    out = STUDY / name / "test_metrics.json"
    if out.exists():  # RESUMABLE: a re-run skips seeds that already finished
        print(f"[seed {seed}] already done -> {out} (skipping)")
        return out

    _register()
    from ultralytics import YOLO

    m = model if Path(model).suffix == ".pt" else str(ROOT / model)
    yolo = YOLO(m)
    if Path(m).suffix == ".yaml":
        yolo.load("yolov8n.pt")  # transfer COCO weights for custom configs
    # Grayscale recipe: kill hue/saturation aug (meaningless on gray), keep brightness (hsv_v).
    # cache=True keeps the tiny NEU-DET in RAM (faster with workers=0; identical results).
    yolo.train(data=str(DATA), imgsz=imgsz, epochs=epochs, seed=seed, optimizer="SGD",
               cos_lr=True, close_mosaic=10, workers=0, device=0, batch=batch, cache=cache,
               hsv_h=0.0, hsv_s=0.0,
               project=str(STUDY), name=name, exist_ok=True, verbose=False)
    # score on TEST
    metrics = yolo.val(data=str(DATA), split="test", imgsz=imgsz, device=0, verbose=False, plots=False)
    out = STUDY / name / "test_metrics.json"
    res = {"tag": tag, "seed": seed, "mAP50": float(metrics.box.map50),
           "mAP50_95": float(metrics.box.map), "precision": float(metrics.box.mp),
           "recall": float(metrics.box.mr)}
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[seed {seed}] TEST mAP50={res['mAP50']:.4f}")
    return out


def collect(tag: str, seeds=SEEDS) -> list[dict]:
    rows = []
    for seed in seeds:
        f = STUDY / f"{tag}_seed{seed}" / "test_metrics.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    return rows


def stats(vals: list[float]) -> dict:
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    ci = T95.get(n - 1, 2.776) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": sd, "ci95": ci,
            "lo": mean - ci, "hi": mean + ci, "values": vals}


def aggregate(tag: str, compare: str | None, seeds=SEEDS):
    a = collect(tag, seeds)
    if not a:
        print(f"No runs for '{tag}'. Train first with --train --tag {tag}."); return
    sa = stats([r["mAP50"] for r in a])
    print(f"\n=== {tag} (n={sa['n']}) TEST mAP@0.5 ===")
    print(f"  mean {sa['mean']:.4f} +/- {sa['std']:.4f} (std)  |  95% CI [{sa['lo']:.4f}, {sa['hi']:.4f}]")
    print(f"  seeds: " + ", ".join(f"{v:.4f}" for v in sa['values']))
    if compare:
        b = collect(compare, seeds)
        if not b:
            print(f"No runs for compare '{compare}'."); return
        sb = stats([r["mAP50"] for r in b])
        print(f"\n=== {compare} (n={sb['n']}) ===")
        print(f"  mean {sb['mean']:.4f} +/- {sb['std']:.4f}  |  95% CI [{sb['lo']:.4f}, {sb['hi']:.4f}]")
        # Welch's t-test (unequal var)
        na, nb = sa['n'], sb['n']
        se = math.sqrt(sa['std']**2/na + sb['std']**2/nb) if na > 1 and nb > 1 else 0
        t = (sb['mean'] - sa['mean']) / se if se else float('nan')
        print(f"\n  delta-mean = {sb['mean']-sa['mean']:+.4f}  Welch t = {t:.2f}")
        sig = abs(t) > 2.776 if se else False
        print(f"  CIs {'OVERLAP -> NOT significant' if (sa['hi']>=sb['lo'] and sb['hi']>=sa['lo']) else 'disjoint'}; "
              f"|t|>2.78 (df~4): {'YES, significant' if sig else 'NO -> treat as a tie'}")
    out = STUDY / f"summary_{tag}.json"
    out.write_text(json.dumps(sa, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="GPU: train 5 seeds for --tag")
    ap.add_argument("--aggregate", action="store_true", help="CPU: mean/std/CI for --tag")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="yolov8n.pt", help=".pt or a configs/*.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--compare", default=None, help="second --tag to two-sample compare against")
    ap.add_argument("--seeds", default=None, help="comma list e.g. 42,123,777 (default: 5 seeds)")
    ap.add_argument("--batch", type=int, default=16, help="batch size (keep fixed across compared arms)")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS

    if args.train:
        STUDY.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            print(f"\n########## TRAIN {args.tag} seed {seed} ##########")
            train_one(args.tag, args.model, args.imgsz, args.epochs, seed, batch=args.batch)
    if args.aggregate or not args.train:
        aggregate(args.tag, args.compare, seeds)


if __name__ == "__main__":
    main()
