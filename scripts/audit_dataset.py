"""
audit_dataset.py - quantitative, evidence-first audit of the NEU-DET YOLO dataset.

Read-only. Computes class balance, image dims, box-size/aspect-ratio distributions,
missing/empty/invalid labels, EXACT duplicates (md5) and cross-split NEAR-duplicates
(dHash Hamming) - the latter is the main way a test mAP can be silently inflated.

    python scripts/audit_dataset.py
Writes docs/audit/data_audit.json and prints a summary.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DROOT = ROOT / "data" / "neu-det-yolo"
SPLITS = ("train", "val", "test")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
OUT = ROOT / "docs" / "audit" / "data_audit.json"


def dhash(pil: Image.Image, n: int = 8) -> int:
    a = np.asarray(pil.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def popcount(x: int) -> int:
    return bin(x).count("1")


def main() -> dict:
    images = {s: sorted((DROOT / "images" / s).glob("*.jpg")) for s in SPLITS}
    rep: dict = {"root": str(DROOT), "splits": {}, "issues": {}}

    md5_map: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, list] = {s: [] for s in SPLITS}  # (stem, dhash, cls_from_name)
    dims = defaultdict(int)
    modes = defaultdict(int)
    box_area, box_ar = [], []
    per_class_area = defaultdict(list)
    cls_img = {s: defaultdict(int) for s in SPLITS}
    cls_inst = {s: defaultdict(int) for s in SPLITS}
    boxes_per_img = []
    missing, empty, invalid = [], [], []
    giant_05 = giant_08 = total_boxes = 0

    for s in SPLITS:
        for img in images[s]:
            stem = img.stem
            raw = img.read_bytes()
            md5_map[hashlib.md5(raw).hexdigest()].append(f"{s}/{stem}")
            with Image.open(img) as im:
                dims[im.size] += 1
                modes[im.mode] += 1
                hashes[s].append((stem, dhash(im)))
                W, H = im.size

            lab = DROOT / "labels" / s / f"{stem}.txt"
            if not lab.exists():
                missing.append(f"{s}/{stem}"); continue
            lines = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()]
            if not lines:
                empty.append(f"{s}/{stem}"); continue
            boxes_per_img.append(len(lines))
            classes_here = set()
            for p in lines:
                if len(p) != 5:
                    invalid.append(f"{s}/{stem}: {p}"); continue
                c, xc, yc, w, h = int(p[0]), *map(float, p[1:])
                if not (0 <= c < len(CLASSES)) or not all(0 <= v <= 1.0001 for v in (xc, yc, w, h)):
                    invalid.append(f"{s}/{stem}: {p}")
                area = w * h
                box_area.append(area)
                box_ar.append(w / h if h > 0 else 0)
                per_class_area[CLASSES[c]].append(area)
                cls_inst[s][CLASSES[c]] += 1
                classes_here.add(CLASSES[c])
                total_boxes += 1
                giant_05 += area > 0.5
                giant_08 += area > 0.8
            for cc in classes_here:
                cls_img[s][cc] += 1

        rep["splits"][s] = {
            "images": len(images[s]),
            "instances_per_class": {c: cls_inst[s][c] for c in CLASSES},
            "images_per_class": {c: cls_img[s][c] for c in CLASSES},
        }

    # exact duplicates (same bytes) within/across splits
    exact_dups = {h: locs for h, locs in md5_map.items() if len(locs) > 1}
    cross_exact = {h: locs for h, locs in exact_dups.items()
                   if len({l.split("/")[0] for l in locs}) > 1}

    # cross-split near-duplicates (dHash Hamming <= THRESH)
    THRESH = 3
    near = {}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        pairs = []
        for sa, ha in hashes[a]:
            for sb, hb in hashes[b]:
                d = popcount(ha ^ hb)
                if d <= THRESH:
                    pairs.append((sa, sb, d))
        near[f"{a}->{b}"] = pairs

    ba = np.array(box_area) if box_area else np.array([0.0])
    rep["image_dims"] = {str(k): v for k, v in dims.items()}
    rep["image_modes"] = dict(modes)
    rep["box_stats"] = {
        "total_boxes": total_boxes,
        "boxes_per_image": {"min": int(min(boxes_per_img)), "mean": round(float(np.mean(boxes_per_img)), 2),
                            "max": int(max(boxes_per_img))} if boxes_per_img else {},
        "area_fraction": {"min": round(float(ba.min()), 3), "median": round(float(np.median(ba)), 3),
                          "mean": round(float(ba.mean()), 3), "max": round(float(ba.max()), 3)},
        "giant_box_pct_gt0.5area": round(100 * giant_05 / max(total_boxes, 1), 1),
        "giant_box_pct_gt0.8area": round(100 * giant_08 / max(total_boxes, 1), 1),
        "per_class_median_area": {c: round(float(np.median(per_class_area[c])), 3)
                                  for c in CLASSES if per_class_area[c]},
        "aspect_ratio_median": round(float(np.median(box_ar)), 2) if box_ar else None,
    }
    rep["issues"] = {
        "missing_labels": missing,
        "empty_labels": empty[:20], "empty_labels_count": len(empty),
        "invalid_labels": invalid[:20], "invalid_labels_count": len(invalid),
        "exact_duplicate_groups": len(exact_dups),
        "cross_split_exact_duplicates": cross_exact,
        "cross_split_near_dup_counts": {k: len(v) for k, v in near.items()},
        "cross_split_near_dup_examples": {k: v[:8] for k, v in near.items()},
        "near_dup_threshold_hamming": THRESH,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # ---- console summary ----
    print(f"images: " + ", ".join(f"{s}={len(images[s])}" for s in SPLITS))
    print(f"dims: {rep['image_dims']}  modes: {rep['image_modes']}")
    bs = rep["box_stats"]
    print(f"boxes: {bs['total_boxes']} total, per-image {bs['boxes_per_image']}")
    print(f"box area frac: {bs['area_fraction']}  | giant >0.5={bs['giant_box_pct_gt0.5area']}% "
          f">0.8={bs['giant_box_pct_gt0.8area']}%")
    print(f"per-class median area: {bs['per_class_median_area']}")
    print("instances/class:")
    for s in SPLITS:
        print(f"  {s}: {rep['splits'][s]['instances_per_class']}")
    iss = rep["issues"]
    print(f"missing={len(missing)} empty={iss['empty_labels_count']} invalid={iss['invalid_labels_count']}")
    print(f"exact dup groups={iss['exact_duplicate_groups']}  cross-split EXACT dups={len(cross_exact)}")
    print(f"cross-split NEAR-dup (Hamming<={THRESH}): {iss['cross_split_near_dup_counts']}")
    if any(near.values()):
        for k, v in near.items():
            if v:
                print(f"  {k} examples: {v[:5]}")
    print(f"\n-> {OUT}")
    return rep


if __name__ == "__main__":
    main()
