"""
check_leakage.py - trustworthy cross-split near-duplicate / leakage check.

The 8x8 dHash in audit_dataset.py is unreliable on flat low-contrast steel textures
(it produced cross-CLASS 'matches', i.e. false positives). This uses a far more
discriminative signature: 32x32 grayscale, per-image z-normalized then L2-normalized,
compared by cosine correlation (== normalized cross-correlation of the pixels).
A real near-duplicate crop scores ~0.97-1.0; unrelated images score near 0.

For each cross-split direction we report, per query image, its single best match in
the other split: the correlation distribution and how many exceed strict thresholds,
plus whether the best match is the SAME class (a real dup should be same-class).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DROOT = ROOT / "data" / "neu-det-yolo"
SPLITS = ("train", "val", "test")


def load(split: str):
    files = sorted((DROOT / "images" / split).glob("*.jpg"))
    vecs, names, cls = [], [], []
    for f in files:
        a = np.asarray(Image.open(f).convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        a = a - a.mean()
        s = a.std()
        a = a / s if s > 1e-6 else np.zeros_like(a)
        n = np.linalg.norm(a)
        a = a / n if n > 1e-6 else a
        vecs.append(a); names.append(f.stem); cls.append(f.stem.rsplit("_", 1)[0])
    return np.array(vecs), names, cls


def report(qa, qn, qc, ba, bn, bc, label):
    sim = qa @ ba.T              # [Nq, Nb] cosine correlation
    best = sim.argmax(axis=1)
    bestsim = sim[np.arange(len(qn)), best]
    same = sum(qc[i] == bc[best[i]] for i in range(len(qn)))
    print(f"\n=== {label}  (queries={len(qn)}, gallery={len(bn)}) ===")
    print(f"best-match corr: min={bestsim.min():.3f} median={np.median(bestsim):.3f} "
          f"max={bestsim.max():.3f}")
    for thr in (0.99, 0.97, 0.95, 0.90):
        n = int((bestsim >= thr).sum())
        print(f"  queries with a twin >= {thr:.2f}: {n}  ({100*n/len(qn):.1f}%)")
    # cross-class fraction among the >=0.95 'matches' (high => artifact, not real dup)
    hi = np.where(bestsim >= 0.95)[0]
    if len(hi):
        cross = sum(qc[i] != bc[best[i]] for i in hi)
        print(f"  of {len(hi)} matches >=0.95, {cross} are CROSS-class "
              f"({100*cross/len(hi):.0f}% -> artifact if high)")
    top = bestsim.argsort()[::-1][:8]
    print("  top pairs:", [(qn[i], bn[best[i]], round(float(bestsim[i]), 3)) for i in top])


def main():
    data = {s: load(s) for s in SPLITS}
    for a, b in (("test", "train"), ("val", "train"), ("test", "val")):
        qa, qn, qc = data[a]
        ba, bn, bc = data[b]
        report(qa, qn, qc, ba, bn, bc, f"{a} queried against {b}")


if __name__ == "__main__":
    main()
