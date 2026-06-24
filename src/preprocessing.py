"""
preprocessing.py - THE canonical preprocessing for the steel-defect models.

Single source of truth, imported by BOTH the inference/eval path and the app
(`src/infer.py`, `src/app.py`, `src/explain.py`). Do not preprocess for the model
anywhere else - if you need model input, call `to_model_input` here.

Why this exists (the RGB->grayscale bug)
----------------------------------------
NEU-DET is a GRAYSCALE dataset. Ultralytics reads each training image with cv2,
which replicates the single gray channel into 3 identical channels (R==G==B).
So the model was trained on *grayscale-replicated* 3-channel tensors.

The old app/XAI path loaded uploads with `Image.open(...).convert("RGB")` and fed
true *colour* to the model - a train/inference mismatch that is invisible on the
(already-gray) NEU-DET files but bites on any colour photo / webcam frame.

`to_model_input` fixes this at the source: convert to single-channel luma, then
replicate to 3 channels, reproducing exactly what training saw. Letterbox and
0-1 normalization are left to Ultralytics' own predict() pipeline (identical to
training), so this function owns only the part that was wrong: the colour->gray
collapse. Because the output has R==G==B, BGR-vs-RGB channel order is irrelevant
downstream.
"""
from __future__ import annotations

import numpy as np

try:  # PIL is the project's image lib; numpy-only fallback keeps the test light
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _luma(rgb: np.ndarray) -> np.ndarray:
    """ITU-R 601 luma of an (H,W,3) uint8 RGB array -> (H,W) uint8.

    Uses PIL's exact fixed-point coefficients (L24: 19595/38470/7471, +0x8000,
    >>16) so PIL `.convert("L")` and the numpy path collapse byte-identically.
    int32 accumulation avoids uint8 overflow (numpy 2.x NEP-50).
    """
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    y = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16
    return y.astype(np.uint8)


def _as_rgb_uint8(img, *, is_bgr: bool) -> np.ndarray:
    """Coerce a PIL image or numpy array to an (H,W,3) uint8 RGB array."""
    if Image is not None and isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        # assume already 0-255 valued; clip defensively for floats
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:  # single-channel gray
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # drop alpha
            arr = arr[:, :, :3]
        if arr.shape[2] == 3:
            return arr[:, :, ::-1].copy() if is_bgr else arr
    raise ValueError(f"Unsupported image shape for preprocessing: {arr.shape}")


def to_model_input(img, *, is_bgr: bool = False) -> np.ndarray:
    """Canonical model input: grayscale-replicated (H,W,3) uint8.

    Args:
        img: a ``PIL.Image`` (assumed RGB) or a numpy array. 2-D arrays are
            treated as single-channel gray; 3-/4-channel arrays are treated as
            RGB unless ``is_bgr=True`` (e.g. a raw cv2 frame).
        is_bgr: set True when passing a BGR numpy array (cv2 convention).

    Returns:
        (H, W, 3) uint8 array with R == G == B (the grayscale the model expects).
        Feed this straight to ``model.predict(...)``; Ultralytics handles the
        letterbox + 0-1 normalization exactly as it did during training.
    """
    rgb = _as_rgb_uint8(img, is_bgr=is_bgr)
    gray = _luma(rgb)
    return np.repeat(gray[:, :, None], 3, axis=2)


def channels_equal(arr: np.ndarray) -> bool:
    """True iff an (H,W,3) array is grayscale-replicated (R == G == B).

    Used by the app to *show* that the grayscale conversion happened.
    """
    arr = np.asarray(arr)
    return arr.ndim == 3 and arr.shape[2] == 3 and bool(
        np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])
    )
