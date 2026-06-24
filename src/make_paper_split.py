"""
Re-split data/neu-det-yolo into the PAPER'S EXACT protocol (s41598-025-93469-5).

Paper: NEU-DET, 1800 imgs, 8:1:1 -> train 1440 / verification(val) 180 / test 180.

NEU-DET's official train folder is already 1440 (240/class) == the paper's train count,
so TRAIN IS KEPT AS-IS. The official 360-image validation pool (60/class) is split
deterministically (seed=42), stratified per class, into 30/class val + 30/class test.

Mapping to Ultralytics: `val` (180) is used during training for early-stop/selection;
`test` (180) is the held-out set for the final reported mAP. This matches the paper's
"verification set (180)" + "test set (180)".

Idempotent: rebuilds val+test from the union of whatever is currently in val+test, so
re-running always yields the same assignment. The train split is never touched.
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLO = ROOT / "data" / "neu-det-yolo"
IMG, LBL = YOLO / "images", YOLO / "labels"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
N_VAL, N_TEST = 30, 30          # per class -> 180 / 180 total
SEED = 42


def stems_in(split):
    d = IMG / split
    return [p.stem for p in d.glob("*.jpg")] if d.exists() else []


def move(stem, dst):
    """Move <stem>.jpg + <stem>.txt into images/<dst> and labels/<dst> from wherever they are."""
    for src in ("val", "test"):
        if src == dst:
            continue
        si, sl = IMG / src / f"{stem}.jpg", LBL / src / f"{stem}.txt"
        if si.exists():
            shutil.move(str(si), str(IMG / dst / f"{stem}.jpg"))
        if sl.exists():
            shutil.move(str(sl), str(LBL / dst / f"{stem}.txt"))


def main():
    for d in (IMG / "val", LBL / "val", IMG / "test", LBL / "test"):
        d.mkdir(parents=True, exist_ok=True)

    pool = set(stems_in("val")) | set(stems_in("test"))   # the 360-image official validation pool
    rng = random.Random(SEED)
    val_sel, test_sel = [], []
    for cls in CLASSES:                                   # fixed class order -> reproducible
        cls_stems = sorted(s for s in pool if s.startswith(cls + "_"))
        rng.shuffle(cls_stems)
        val_sel += cls_stems[:N_VAL]
        test_sel += cls_stems[N_VAL:N_VAL + N_TEST]
        leftover = cls_stems[N_VAL + N_TEST:]
        assert not leftover, f"{cls}: expected 60 in pool, got {len(cls_stems)}"

    for stem in val_sel:
        move(stem, "val")
    for stem in test_sel:
        move(stem, "test")

    # data.yaml with the test key (Ultralytics resolves 'path' as the dataset root)
    (YOLO / "data.yaml").write_text(
        f"path: {YOLO.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 6\n"
        "names:\n"
        "  0: crazing\n"
        "  1: inclusion\n"
        "  2: patches\n"
        "  3: pitted_surface\n"
        "  4: rolled-in_scale\n"
        "  5: scratches\n"
    )

    # report
    print("Paper 8:1:1 split written to", YOLO)
    for split in ("train", "val", "test"):
        n = len(stems_in(split)) if split != "train" else len(list((IMG / "train").glob("*.jpg")))
        per = {c: len([s for s in stems_in(split) if s.startswith(c + "_")]) for c in CLASSES} if split != "train" else {}
        extra = "  " + "  ".join(f"{c.split('_')[0]}:{n}" for c, n in per.items()) if per else ""
        print(f"  {split:5s}: {n} images{extra}")


if __name__ == "__main__":
    main()
