"""Turn ORM objects into API response dicts (with /media URLs + class counts)."""
from __future__ import annotations

from ..models import Image, Inspection


def _media(path: str) -> str:
    return f"/media/{path}" if path else ""


def detection_out(d) -> dict:
    return {"id": d.id, "cls_name": d.cls_name, "confidence": d.confidence,
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2}


def image_out(img: Image) -> dict:
    return {
        "id": img.id,
        "filename": img.filename,
        "original_url": _media(img.original_path),
        "annotated_url": _media(img.annotated_path),
        "cam_url": _media(img.cam_path),
        "width": img.width,
        "height": img.height,
        "n_defects": img.n_defects,
        "brightness": img.brightness,
        "quality": img.quality,
        "detections": [detection_out(d) for d in img.detections],
    }


def inspection_out(insp: Inspection) -> dict:
    return {
        "id": insp.id, "title": insp.title, "mode": insp.mode, "conf": insp.conf,
        "imgsz": insp.imgsz, "status": insp.status, "n_images": insp.n_images,
        "n_defects": insp.n_defects, "created_at": insp.created_at,
    }


def inspection_detail(insp: Inspection) -> dict:
    counts: dict[str, int] = {}
    for d in insp.detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    return {
        **inspection_out(insp),
        "images": [image_out(i) for i in insp.images],
        "class_counts": counts,
    }
