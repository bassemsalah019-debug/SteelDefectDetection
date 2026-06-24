"""
voc_to_yolo.py
Convert NEU-DET Pascal VOC XML annotations to YOLO txt format.

YOLO format per line:  <class_id> <x_center> <y_center> <width> <height>
All coordinates normalized to [0, 1] by image width/height.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# NEU-DET class order — MUST match configs/neu_det.yaml
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Some NEU-DET copies use abbreviated tags — map them to canonical names.
NAME_ALIASES = {
    "cr": "crazing",
    "in": "inclusion",
    "pa": "patches",
    "ps": "pitted_surface",
    "rs": "rolled-in_scale",
    "sc": "scratches",
}


def normalize_class_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    return NAME_ALIASES.get(name, name)


def convert_box(size, box):
    """Convert (xmin, xmax, ymin, ymax) pixels to YOLO normalized xywh."""
    w_img, h_img = size
    xmin, xmax, ymin, ymax = box
    x_center = (xmin + xmax) / 2.0 / w_img
    y_center = (ymin + ymax) / 2.0 / h_img
    width = (xmax - xmin) / w_img
    height = (ymax - ymin) / h_img
    return x_center, y_center, width, height


def convert_annotation(xml_path: Path, out_txt_path: Path) -> int:
    """Convert one XML file to one YOLO txt file. Returns number of boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    w_img = int(size_node.find("width").text)
    h_img = int(size_node.find("height").text)

    lines = []
    for obj in root.iter("object"):
        raw_name = obj.find("name").text
        cls_name = normalize_class_name(raw_name)
        if cls_name not in CLASSES:
            print(f"  [skip] unknown class '{raw_name}' in {xml_path.name}")
            continue
        cls_id = CLASSES.index(cls_name)

        bnd = obj.find("bndbox")
        box = (
            float(bnd.find("xmin").text),
            float(bnd.find("xmax").text),
            float(bnd.find("ymin").text),
            float(bnd.find("ymax").text),
        )
        x, y, w, h = convert_box((w_img, h_img), box)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(lines))
    return len(lines)


def convert_folder(xml_dir: str, out_dir: str) -> None:
    """Convert all XML files in a folder."""
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    xml_files = sorted(xml_dir.glob("*.xml"))
    total_boxes = 0
    for xml_path in xml_files:
        out_txt = out_dir / (xml_path.stem + ".txt")
        total_boxes += convert_annotation(xml_path, out_txt)
    print(f"Converted {len(xml_files)} files, {total_boxes} boxes -> {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert VOC XML to YOLO txt")
    parser.add_argument("--xml_dir", required=True, help="Folder with .xml files")
    parser.add_argument("--out_dir", required=True, help="Output folder for .txt")
    args = parser.parse_args()
    convert_folder(args.xml_dir, args.out_dir)
