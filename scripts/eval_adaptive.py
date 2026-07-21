"""
eval_adaptive.py - Fixed vs Adaptive confidence thresholding, measured on TEST.

Self-contained detection evaluator (does NOT depend on Ultralytics' val internals,
because we need to apply a *per-detection, per-class* threshold policy that the
stock validator cannot express). For one model it:

  1. runs ONE forward pass per image at a low candidate floor (so both policies
     score on the SAME candidate pool - a fair comparison);
  2. applies each policy
        FIXED    : keep conf >= --conf
        ADAPTIVE : keep conf >= T(class, image)   (src/adaptive_threshold.py)
  3. greedily matches detections to ground truth at IoU 0.5 and reports, per policy:
        mAP@0.5 (VOC all-points AP over the retained, ranked detections),
        Precision, Recall, F1 (macro = mean over classes, the project's convention),
        plus per-class AP / P / R;
  4. separately times the two REAL code paths (fixed `predict` vs `predict_adaptive`)
     over the test set to report FPS / latency and the adaptive overhead.

All numbers are written to a JSON and printed as a comparison table.

    python scripts/eval_adaptive.py --weights results/baseline_640/weights/best.pt \
        --imgsz 640 --conf 0.25 --device 0

Notes
-----
* mAP here is *policy-restricted* (computed on detections the policy keeps). The
  unrestricted ceiling (conf -> 0) is the project's headline 0.7525; thresholding
  is an OPERATING-POINT tool, so the honest comparison is Precision/Recall/F1 at
  the deployment threshold. See docs/audit/ADAPTIVE_THRESHOLDING.md.
* GPU job: per the project rule the human runs it. It imports cleanly without a GPU
  and falls back to CPU if --device cpu.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adaptive_threshold import (  # noqa: E402
    CLASS_NAMES,
    AdaptiveConfig,
    AdaptiveThresholder,
    compute_image_signals,
    keep_mask,
    thresholds_by_id,
)
from src.infer import predict as predict_fixed, predict_adaptive  # noqa: E402
from src.preprocessing import to_model_input  # noqa: E402

NC = len(CLASS_NAMES)
EVAL_FLOOR = 0.01  # gather candidates this low so the AP "ceiling" is faithful


# --------------------------------------------------------------------------- #
# Geometry + AP
# --------------------------------------------------------------------------- #
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a [N,4] and b [M,4] (xyxy) -> [N,M]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC all-points AP: area under the precision envelope."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match_image(dets, gts, iou_thr=0.5):
    """Greedy per-class matching for one image.

    dets: (xyxy [N,4], conf [N], cls [N]); gts: (xyxy [M,4], cls [M]).
    Returns per class c:
        scored[c] -> list of (conf, is_tp)   (for AP, dets ranked later)
        n_tp[c], n_det[c], n_pos[c]          (for operating-point P/R/F1)
    """
    dx, dconf, dcls = dets
    gx, gcls = gts
    scored = {c: [] for c in range(NC)}
    n_tp = {c: 0 for c in range(NC)}
    n_det = {c: 0 for c in range(NC)}
    n_pos = {c: 0 for c in range(NC)}

    for c in range(NC):
        gi = np.where(gcls == c)[0]
        n_pos[c] += len(gi)
        di = np.where(dcls == c)[0]
        n_det[c] += len(di)
        if len(di) == 0:
            continue
        order = di[np.argsort(-dconf[di])]  # high conf first
        gboxes = gx[gi]
        taken = np.zeros(len(gi), dtype=bool)
        ious = _iou_matrix(dx[order], gboxes) if len(gi) else np.zeros((len(order), 0))
        for r, d in enumerate(order):
            tp = 0
            if len(gi):
                j = int(np.argmax(ious[r])) if ious.shape[1] else -1
                if j >= 0 and ious[r, j] >= iou_thr and not taken[j]:
                    taken[j] = True
                    tp = 1
            scored[c].append((float(dconf[d]), tp))
            n_tp[c] += tp
    return scored, n_tp, n_det, n_pos


def _aggregate(per_image):
    """Combine per-image match results into per-class AP / P / R / F1 + macro."""
    scored = {c: [] for c in range(NC)}
    n_tp = {c: 0 for c in range(NC)}
    n_det = {c: 0 for c in range(NC)}
    n_pos = {c: 0 for c in range(NC)}
    for s, tp, det, pos in per_image:
        for c in range(NC):
            scored[c].extend(s[c])
            n_tp[c] += tp[c]
            n_det[c] += det[c]
            n_pos[c] += pos[c]

    per_class = {}
    aps, precs, recs, f1s = [], [], [], []
    tot_tp = tot_fp = tot_pos = 0
    for c in range(NC):
        name = CLASS_NAMES[c]
        # AP from ranked detections
        if scored[c] and n_pos[c] > 0:
            arr = sorted(scored[c], key=lambda x: -x[0])
            tps = np.array([t for _, t in arr], dtype=np.float64)
            fps = 1.0 - tps
            ctp, cfp = np.cumsum(tps), np.cumsum(fps)
            rec = ctp / max(n_pos[c], 1)
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            ap = _voc_ap(rec, prec)
        else:
            ap = 0.0
        # operating-point P/R/F1
        tp = n_tp[c]; fp = n_det[c] - tp; pos = n_pos[c]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / pos if pos else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[name] = {"AP50": round(ap, 4), "P": round(p, 4),
                           "R": round(r, 4), "F1": round(f1, 4),
                           "TP": tp, "FP": fp, "n_gt": pos}
        aps.append(ap); precs.append(p); recs.append(r); f1s.append(f1)
        tot_tp += tp; tot_fp += fp; tot_pos += pos

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    micro_r = tot_tp / tot_pos if tot_pos else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "mAP50": round(float(np.mean(aps)), 4),
        "precision_macro": round(float(np.mean(precs)), 4),
        "recall_macro": round(float(np.mean(recs)), 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
        "precision_micro": round(micro_p, 4),
        "recall_micro": round(micro_r, 4),
        "f1_micro": round(micro_f1, 4),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load_gt(label_path: Path, w: int, h: int):
    """YOLO txt (normalized cx cy w h) -> (xyxy [M,4] pixels, cls [M])."""
    if not label_path.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), int)
    boxes, classes = [], []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2]); classes.append(c)
    return (np.array(boxes, np.float32).reshape(-1, 4),
            np.array(classes, int))


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
def evaluate(weights: str, images_dir: Path, labels_dir: Path, *,
             imgsz: int = 640, conf_fixed: float = 0.25, device="0",
             config: AdaptiveConfig | None = None) -> dict:
    from PIL import Image
    from ultralytics import YOLO
    try:
        from src.modules import register, register_lzy
        register(verbose=False); register_lzy(verbose=False)
    except Exception:
        pass

    cfg = config or AdaptiveConfig()
    at = AdaptiveThresholder(config=cfg)
    model = YOLO(str(ROOT / weights if not Path(weights).is_absolute() else weights))

    imgs = sorted(p for p in images_dir.glob("*.jpg"))
    fixed_pi, adapt_pi = [], []

    for ip in imgs:
        pil = Image.open(ip).convert("RGB")
        w, h = pil.size
        model_input = to_model_input(pil)
        res = model.predict(model_input, conf=EVAL_FLOOR, imgsz=imgsz,
                            device=device, verbose=False)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            dx = np.zeros((0, 4), np.float32); dconf = np.zeros((0,)); dcls = np.zeros((0,), int)
        else:
            dx = b.xyxy.detach().cpu().numpy()
            dconf = b.conf.detach().cpu().numpy()
            dcls = b.cls.detach().cpu().numpy().astype(int)
        gx, gcls = _load_gt(labels_dir / (ip.stem + ".txt"), w, h)

        # FIXED policy
        mf = dconf >= conf_fixed
        fixed_pi.append(_match_image((dx[mf], dconf[mf], dcls[mf]), (gx, gcls)))

        # ADAPTIVE policy (density from candidates above the live floor)
        density = int((dconf >= cfg.candidate_floor).sum())
        sig = compute_image_signals(model_input, density=density, config=cfg)
        thr = thresholds_by_id(sig, cfg)
        ma = keep_mask(dcls, dconf, thr)
        adapt_pi.append(_match_image((dx[ma], dconf[ma], dcls[ma]), (gx, gcls)))

    fixed = _aggregate(fixed_pi)
    adaptive = _aggregate(adapt_pi)

    timing = _benchmark(model, imgs, imgsz, conf_fixed, device, at)

    return {
        "weights": str(weights), "imgsz": imgsz, "conf_fixed": conf_fixed,
        "device": _device_name(device), "n_images": len(imgs),
        "eval_candidate_floor": EVAL_FLOOR,
        "adaptive_config": _cfg_dict(cfg),
        "fixed": fixed, "adaptive": adaptive, "timing": timing,
    }


def _benchmark(model, imgs, imgsz, conf_fixed, device, at, warmup=8):
    """Time the two REAL code paths (incl. grayscale preprocessing + overhead)."""
    from PIL import Image

    sample = [Image.open(p).convert("RGB") for p in imgs[: min(len(imgs), 80)]]
    for im in sample[:warmup]:  # warmup
        predict_fixed(model, im, conf=conf_fixed, imgsz=imgsz, device=device, verbose=False)
        predict_adaptive(model, im, imgsz=imgsz, device=device, thresholder=at)

    def timed(fn):
        t0 = time.perf_counter()
        for im in sample:
            fn(im)
        dt = time.perf_counter() - t0
        return dt / len(sample)

    lat_fixed = timed(lambda im: predict_fixed(model, im, conf=conf_fixed,
                                               imgsz=imgsz, device=device, verbose=False))
    lat_adapt = timed(lambda im: predict_adaptive(model, im, imgsz=imgsz,
                                                  device=device, thresholder=at))
    return {
        "fixed_latency_ms": round(lat_fixed * 1000, 3),
        "adaptive_latency_ms": round(lat_adapt * 1000, 3),
        "overhead_ms": round((lat_adapt - lat_fixed) * 1000, 3),
        "fixed_fps": round(1.0 / lat_fixed, 1) if lat_fixed else None,
        "adaptive_fps": round(1.0 / lat_adapt, 1) if lat_adapt else None,
        "n_timed": len(sample),
    }


def _cfg_dict(cfg: AdaptiveConfig) -> dict:
    return {"t0": cfg.t0, "k_class": cfg.k_class, "w_bright": cfg.w_bright,
            "w_quality": cfg.w_quality, "w_density": cfg.w_density,
            "t_min": cfg.t_min, "t_max": cfg.t_max,
            "candidate_floor": cfg.candidate_floor}


def _device_name(device) -> str:
    try:
        import torch
        if str(device) not in ("cpu", "-1", "None") and torch.cuda.is_available():
            return f"cuda:{device} ({torch.cuda.get_device_name(int(device))})"
    except Exception:
        pass
    return "cpu"


def _print_table(r: dict) -> None:
    f, a, t = r["fixed"], r["adaptive"], r["timing"]
    print(f"\n=== Fixed vs Adaptive thresholding — {r['weights']} ===")
    print(f"{r['n_images']} TEST imgs @ imgsz {r['imgsz']} on {r['device']} "
          f"| fixed conf {r['conf_fixed']}\n")
    rows = [
        ("mAP@0.5 (policy-restricted)", f["mAP50"], a["mAP50"]),
        ("Precision (macro)", f["precision_macro"], a["precision_macro"]),
        ("Recall (macro)", f["recall_macro"], a["recall_macro"]),
        ("F1 (macro)", f["f1_macro"], a["f1_macro"]),
        ("Precision (micro)", f["precision_micro"], a["precision_micro"]),
        ("Recall (micro)", f["recall_micro"], a["recall_micro"]),
        ("F1 (micro)", f["f1_micro"], a["f1_micro"]),
        ("Latency (ms)", t["fixed_latency_ms"], t["adaptive_latency_ms"]),
        ("FPS", t["fixed_fps"], t["adaptive_fps"]),
    ]
    print(f"{'metric':<30}{'Fixed':>12}{'Adaptive':>12}{'Δ':>12}")
    for name, fv, av in rows:
        d = (av - fv) if isinstance(fv, (int, float)) and isinstance(av, (int, float)) else ""
        ds = f"{d:+.4f}" if isinstance(d, float) else ""
        print(f"{name:<30}{fv:>12}{av:>12}{ds:>12}")
    print(f"\nAdaptive overhead: {t['overhead_ms']:+.3f} ms/img "
          f"({t['n_timed']} imgs timed)\n")
    print(f"{'class':<16}{'AP fix':>8}{'AP ada':>8}{'R fix':>8}{'R ada':>8}{'F1 fix':>8}{'F1 ada':>8}")
    for c in CLASS_NAMES:
        fc, ac = f["per_class"][c], a["per_class"][c]
        print(f"{c:<16}{fc['AP50']:>8}{ac['AP50']:>8}{fc['R']:>8}{ac['R']:>8}"
              f"{fc['F1']:>8}{ac['F1']:>8}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Fixed vs Adaptive thresholding eval")
    ap.add_argument("--weights", default="results/baseline_640/weights/best.pt")
    ap.add_argument("--images", default=str(ROOT / "data/neu-det-yolo/images/test"))
    ap.add_argument("--labels", default=str(ROOT / "data/neu-det-yolo/labels/test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="fixed-mode confidence")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(ROOT / "docs/audit/adaptive_eval.json"))
    args = ap.parse_args()

    r = evaluate(args.weights, Path(args.images), Path(args.labels),
                 imgsz=args.imgsz, conf_fixed=args.conf, device=args.device)
    Path(args.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
    _print_table(r)
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    _cli()
