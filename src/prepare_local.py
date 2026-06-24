"""
prepare_local.py — Build the YOLOv8 dataset from the LOCAL NEU-DET copy.

No Kaggle download required: this reads the raw Pascal-VOC data already sitting
in  data/NEU-DET/{train,validation}/  and produces the flat YOLO layout
Ultralytics expects:

    data/neu-det-yolo/
        images/train/  images/val/
        labels/train/  labels/val/
        data.yaml

It uses the OFFICIAL NEU-DET split (the train/ and validation/ folders) rather
than a random re-split, so results are stratified per class and reproducible,
and comparable to published NEU-DET benchmarks.

Run from anywhere:
    python src/prepare_local.py
"""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "NEU-DET"
OUT = ROOT / "data" / "neu-det-yolo"

# Map raw split folder -> YOLO split name
SPLITS = {"train": "train", "validation": "val"}

# --- Classes (MUST match configs/neu_det.yaml) -------------------------------
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
# Some NEU-DET copies use abbreviated tags — map them to canonical names.
ALIASES = {
    "cr": "crazing", "in": "inclusion", "pa": "patches",
    "ps": "pitted_surface", "rs": "rolled-in_scale", "sc": "scratches",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def norm_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    return ALIASES.get(name, name)


def xml_to_yolo_lines(xml_path: Path) -> list[str]:
    """Convert one VOC XML to YOLO 'cls xc yc w h' lines (normalized)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    lines = []
    for obj in root.iter("object"):
        name = norm_name(obj.find("name").text)
        if name not in CLASSES:
            print(f"  [skip] unknown class '{name}' in {xml_path.name}")
            continue
        cid = CLASSES.index(name)
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text); xmax = float(b.find("xmax").text)
        ymin = float(b.find("ymin").text); ymax = float(b.find("ymax").text)
        xc = (xmin + xmax) / 2 / W
        yc = (ymin + ymax) / 2 / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def index_annotations() -> dict[str, Path]:
    """Index every .xml by filename stem, wherever it lives (handles any
    misplaced annotations across the train/validation folders)."""
    idx = {}
    for xml in RAW.rglob("*.xml"):
        idx[xml.stem] = xml
    return idx


def main() -> None:
    assert RAW.exists(), f"Raw data not found: {RAW}"
    ann_index = index_annotations()
    print(f"Indexed {len(ann_index)} annotations under {RAW}")

    # Fresh output tree
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    totals = {}
    missing = []
    for raw_split, yolo_split in SPLITS.items():
        img_root = RAW / raw_split / "images"
        n_imgs = n_boxes = 0
        for img in sorted(img_root.rglob("*")):
            if img.suffix.lower() not in IMG_EXT:
                continue
            xml = ann_index.get(img.stem)
            if xml is None:
                missing.append(img.name)
                continue
            shutil.copy(img, OUT / f"images/{yolo_split}" / (img.stem + img.suffix))
            lines = xml_to_yolo_lines(xml)
            (OUT / f"labels/{yolo_split}" / (img.stem + ".txt")).write_text("\n".join(lines))
            n_imgs += 1
            n_boxes += len(lines)
        totals[yolo_split] = (n_imgs, n_boxes)
        print(f"[{yolo_split:5}] {n_imgs} images, {n_boxes} boxes")

    if missing:
        print(f"WARNING: {len(missing)} images had no annotation: {missing[:10]}")

    # Write the YOLO data config (absolute path = robust under Ultralytics)
    yaml_text = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 6\n"
        "names:\n"
        "  0: crazing\n"
        "  1: inclusion\n"
        "  2: patches\n"
        "  3: pitted_surface\n"
        "  4: rolled-in_scale\n"
        "  5: scratches\n"
    )
    (OUT / "data.yaml").write_text(yaml_text)
    print(f"\nWrote {OUT / 'data.yaml'}")
    print("Dataset ready at:", OUT)


if __name__ == "__main__":
    main()
