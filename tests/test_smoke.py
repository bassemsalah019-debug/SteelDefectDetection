"""
Smoke tests - reproducibility + sanity guardrails.

Run from the project root with the project venv:
    python -m pytest tests/ -q

Tests skip gracefully when the dataset isn't built or ultralytics isn't installed,
so they never block a fresh checkout.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def test_split_counts():
    """Dataset must follow the paper's 8:1:1 split: 1440 / 180 / 180."""
    base = ROOT / "data" / "neu-det-yolo" / "images"
    if not base.exists():
        pytest.skip("dataset not built (run 01_data_preparation.ipynb)")
    counts = {s: len(list((base / s).glob("*.jpg"))) for s in ("train", "val", "test")}
    assert counts == {"train": 1440, "val": 180, "test": 180}, counts


def test_val_test_class_balance():
    """val and test must be 30 imgs per class (stratified)."""
    base = ROOT / "data" / "neu-det-yolo" / "images"
    if not base.exists():
        pytest.skip("dataset not built")
    for split in ("val", "test"):
        for c in CLASSES:
            n = len(list((base / split).glob(f"{c}_*.jpg")))
            assert n == 30, f"{split}/{c}: expected 30, got {n}"


def test_improved_model_is_lighter_than_baseline():
    """The improved architecture must build and be lighter than the 3.01M baseline."""
    try:
        from ultralytics import YOLO

        from src.modules import register
    except Exception:
        pytest.skip("ultralytics not installed")
    register()
    model = YOLO(str(ROOT / "configs" / "yolov8n_improved.yaml"))
    n_params = sum(p.numel() for p in model.model.parameters())
    assert n_params < 3_000_000, f"expected < 3.0M params, got {n_params:,}"


def test_explain_api_imports():
    """The XAI module must expose its public API without importing torch-heavy deps eagerly failing."""
    from src.explain import EigenCAM, enable_custom_modules, overlay_cam  # noqa: F401

    assert callable(enable_custom_modules)
    assert callable(overlay_cam)


def test_eigencam_projection_shape():
    """_get_2d_projection reduces [C,H,W] -> [H,W]."""
    import numpy as np

    from src.explain import _get_2d_projection

    proj = _get_2d_projection(np.random.rand(16, 8, 8).astype("float32"))
    assert proj.shape == (8, 8)


def test_attention_summary_locates_hot_region():
    """attention_summary names the dominant region + coverage %."""
    import numpy as np

    from src.explain import attention_summary

    heat = np.zeros((20, 20), dtype="float32")
    heat[:6, :6] = 1.0  # bright top-left
    s = attention_summary(heat)
    assert "top-left" in s and "%" in s


def test_attention_summary_counts_aligned_detections():
    """Detections whose centre lands in the hot region are counted."""
    import numpy as np

    from src.explain import attention_summary
    from src.report import Detection

    heat = np.zeros((20, 20), dtype="float32")
    heat[:10, :10] = 1.0  # hot top-left quadrant
    dets = [Detection("scratches", 0.9, [0, 0, 40, 40]),      # centre in hot zone
            Detection("inclusion", 0.8, [160, 160, 200, 200])]  # centre in cold zone
    s = attention_summary(heat, dets, img_size=(200, 200))
    assert "1 of 2 detections" in s
