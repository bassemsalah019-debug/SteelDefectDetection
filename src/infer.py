"""
infer.py - the one inference wrapper shared by the app and the eval harness.

Both surfaces (Streamlit app, evaluation/inference scripts) MUST go through here
so they preprocess identically. `preprocess` is a thin alias of the canonical
`to_model_input` (see src/preprocessing.py) - kept as a named entry point so the
parity test can assert the app path and the inference path are byte-identical.

Nothing torch/ultralytics-heavy is imported at module load, so this stays cheap
to import (the parity test imports it without a model).
"""
from __future__ import annotations

from src.preprocessing import to_model_input

__all__ = ["preprocess", "predict"]


def preprocess(img, *, is_bgr: bool = False):
    """Canonical preprocessing entry point for the inference/app path."""
    return to_model_input(img, is_bgr=is_bgr)


def predict(model, img, *, is_bgr: bool = False, **predict_kwargs):
    """Run YOLO detection on the canonically-preprocessed (grayscale) input.

    Returns the single ``ultralytics.engine.results.Results`` for the image.
    Any extra kwargs (conf, iou, imgsz, verbose, ...) pass through to predict().
    """
    model_input = preprocess(img, is_bgr=is_bgr)
    return model.predict(model_input, **predict_kwargs)[0]
