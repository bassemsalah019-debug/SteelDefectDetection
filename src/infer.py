"""
infer.py - the one inference wrapper shared by the app and the eval harness.

Both surfaces (Streamlit app, evaluation/inference scripts) MUST go through here
so they preprocess identically. `preprocess` is a thin alias of the canonical
`to_model_input` (see src/preprocessing.py) - kept as a named entry point so the
parity test can assert the app path and the inference path are byte-identical.

Two inference modes are exposed:
  - FIXED threshold:    `predict(...)` -> one global conf (the classic behaviour).
  - ADAPTIVE threshold: `predict_adaptive(...)` -> per-class, per-image conf from
                        src/adaptive_threshold.py (post-processing only; the model
                        graph and ONNX export are untouched).

Nothing torch/ultralytics-heavy is imported at module load, so this stays cheap
to import (the parity test imports it without a model). adaptive_threshold is
numpy-only, so importing it here keeps that guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.adaptive_threshold import (
    AdaptiveConfig,
    AdaptiveThresholder,
    ImageSignals,
    keep_mask,
)
from src.preprocessing import to_model_input

__all__ = ["preprocess", "predict", "predict_adaptive", "AdaptiveResult"]


def preprocess(img, *, is_bgr: bool = False):
    """Canonical preprocessing entry point for the inference/app path."""
    return to_model_input(img, is_bgr=is_bgr)


def predict(model, img, *, is_bgr: bool = False, **predict_kwargs):
    """Run YOLO detection on the canonically-preprocessed (grayscale) input.

    FIXED-threshold mode: the caller's ``conf`` (if any) is a single global value.
    Returns the single ``ultralytics.engine.results.Results`` for the image.
    Any extra kwargs (conf, iou, imgsz, verbose, ...) pass through to predict().
    """
    model_input = preprocess(img, is_bgr=is_bgr)
    return model.predict(model_input, **predict_kwargs)[0]


@dataclass
class AdaptiveResult:
    """Output of the adaptive path: the (already filtered) detections plus the
    signals / thresholds that produced them (for display + the report)."""
    result: Any                      # ultralytics Results, filtered to kept boxes
    signals: ImageSignals
    thresholds_by_id: dict[int, float]
    detail: dict                     # AdaptiveThresholder.describe(...) breakdown


def _filter_result(result, mask: np.ndarray):
    """Index an Ultralytics Results object by a boolean keep-mask (torch, on-device)."""
    import torch

    device = result.boxes.data.device if result.boxes is not None else "cpu"
    tmask = torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool, device=device)
    return result[tmask]


def predict_adaptive(model, img, *, is_bgr: bool = False,
                     config: AdaptiveConfig | None = None,
                     thresholder: AdaptiveThresholder | None = None,
                     **predict_kwargs) -> AdaptiveResult:
    """Run YOLO detection with ADAPTIVE per-class confidence thresholds.

    Pipeline (post-processing only - the model and any ONNX/TRT export are
    untouched):
      1. canonical grayscale preprocessing (identical to the fixed path);
      2. gather candidates at the low ``candidate_floor`` conf;
      3. compute cheap image signals (brightness/contrast/sharpness/density);
      4. derive a per-class threshold and keep ``conf >= T(class, image)``.

    Any ``conf`` in predict_kwargs is ignored (adaptive owns the threshold); other
    kwargs (imgsz, iou, device, verbose, ...) pass straight through.
    """
    at = thresholder or AdaptiveThresholder(config=config)
    cfg = at.config
    predict_kwargs.pop("conf", None)  # adaptive owns confidence

    model_input = preprocess(img, is_bgr=is_bgr)
    result = model.predict(model_input, conf=cfg.candidate_floor, **predict_kwargs)[0]

    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    signals = at.signals(model_input, density=n)
    thr_by_id = at.thresholds_by_id(signals)
    detail = at.describe(signals)

    if n == 0:
        return AdaptiveResult(result, signals, thr_by_id, detail)

    cls_ids = boxes.cls.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    mask = keep_mask(cls_ids, confs, thr_by_id)
    return AdaptiveResult(_filter_result(result, mask), signals, thr_by_id, detail)
