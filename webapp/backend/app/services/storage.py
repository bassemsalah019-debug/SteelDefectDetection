"""Local file storage for uploaded + generated images (served at /media)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from ..config import get_settings

settings = get_settings()
_ROOT = Path(settings.upload_dir)


def save_image(pil: PILImage.Image, inspection_id: str, name: str) -> str:
    """Save a PIL image under <uploads>/<inspection_id>/<name>; return its web path."""
    folder = _ROOT / inspection_id
    folder.mkdir(parents=True, exist_ok=True)
    pil.convert("RGB").save(folder / name, format="JPEG", quality=90)
    return f"{inspection_id}/{name}"


def delete_inspection_files(inspection_id: str) -> None:
    """Remove all stored files for an inspection (best-effort)."""
    import shutil

    folder = _ROOT / inspection_id
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
