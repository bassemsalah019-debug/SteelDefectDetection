"""
Unit tests for adaptive confidence thresholding (src/adaptive_threshold.py).

Light by design: numpy only, no torch/ultralytics/streamlit, so they run in CI on a
fresh checkout (same contract as test_preprocessing_parity.py).

    python -m pytest tests/test_adaptive_threshold.py -q
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adaptive_threshold import (  # noqa: E402
    CLASS_NAMES,
    AdaptiveConfig,
    AdaptiveThresholder,
    adaptive_thresholds,
    class_base_thresholds,
    compute_image_signals,
    global_adjustment,
    keep_mask,
    thresholds_by_id,
)

CFG = AdaptiveConfig()


# ---- class difficulty anchors -------------------------------------------------
def test_base_threshold_orders_by_difficulty():
    """Harder (lower-AP) classes get a LOWER base threshold than easy ones."""
    base = class_base_thresholds(CFG)
    assert base["crazing"] < base["scratches"] < base["patches"], base
    assert base["crazing"] < CFG.t0 < base["patches"]


def test_base_thresholds_within_bounds():
    base = class_base_thresholds(CFG)
    for v in base.values():
        assert CFG.t_min <= v <= CFG.t_max


# ---- signals ------------------------------------------------------------------
def test_flat_image_has_zero_contrast_and_low_quality():
    flat = np.full((128, 128, 3), 120, dtype=np.uint8)
    sig = compute_image_signals(flat, density=2, config=CFG)
    assert abs(sig.brightness - 120 / 255) < 1e-3
    assert sig.contrast == 0.0
    assert sig.sharpness == 0.0
    assert sig.quality < 0.05  # no contrast, no sharpness -> poor quality


def test_textured_image_has_higher_quality_than_flat():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    flat = np.full((128, 128), 120, dtype=np.uint8)
    assert compute_image_signals(noisy).quality > compute_image_signals(flat).quality


def test_grayscale_replicated_uses_channel0():
    """A gray-replicated (R==G==B) array must read as its single channel, cheaply."""
    g = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    rep = np.repeat(g[:, :, None], 3, axis=2)
    assert compute_image_signals(rep).brightness == compute_image_signals(g).brightness


# ---- global adjustments -------------------------------------------------------
def test_poor_brightness_lowers_threshold():
    bright_ideal = compute_image_signals(np.full((64, 64), 128, np.uint8), density=3, config=CFG)
    dark = compute_image_signals(np.full((64, 64), 10, np.uint8), density=3, config=CFG)
    # the dark frame's brightness penalty must push its total adjustment more negative
    assert global_adjustment(dark, CFG)["brightness"] < global_adjustment(bright_ideal, CFG)["brightness"]


def test_density_raises_threshold():
    sig_sparse = compute_image_signals(np.full((64, 64), 128, np.uint8), density=1, config=CFG)
    sig_crowd = compute_image_signals(np.full((64, 64), 128, np.uint8), density=9, config=CFG)
    assert global_adjustment(sig_crowd, CFG)["density"] > global_adjustment(sig_sparse, CFG)["density"]


# ---- final thresholds ---------------------------------------------------------
def test_thresholds_keys_and_bounds():
    sig = compute_image_signals(np.full((64, 64), 30, np.uint8), density=12, config=CFG)
    by_name = adaptive_thresholds(sig, CFG)
    by_id = thresholds_by_id(sig, CFG)
    assert set(by_name) == set(CLASS_NAMES)
    assert set(by_id) == set(range(len(CLASS_NAMES)))
    for v in by_name.values():
        assert CFG.t_min <= v <= CFG.t_max


def test_hard_class_keeps_lower_threshold_than_easy():
    """Per-class ordering survives the global shift (same shift for all classes)."""
    sig = compute_image_signals(np.full((64, 64), 60, np.uint8), density=4, config=CFG)
    thr = adaptive_thresholds(sig, CFG)
    assert thr["crazing"] < thr["patches"]


# ---- mask ---------------------------------------------------------------------
def test_keep_mask_applies_per_class_threshold():
    thr = {0: 0.15, 1: 0.40}  # crazing low, inclusion high
    cls = np.array([0, 0, 1, 1])
    conf = np.array([0.20, 0.10, 0.30, 0.50])
    mask = keep_mask(cls, conf, thr)
    # crazing 0.20>=0.15 keep, 0.10<0.15 drop; inclusion 0.30<0.40 drop, 0.50>=0.40 keep
    assert mask.tolist() == [True, False, False, True]


def test_keep_mask_empty():
    assert keep_mask(np.array([]), np.array([]), {0: 0.2}).tolist() == []


# ---- facade -------------------------------------------------------------------
def test_thresholder_describe_is_complete():
    at = AdaptiveThresholder()
    sig = at.signals(np.full((80, 80), 100, np.uint8), density=3)
    d = at.describe(sig)
    assert set(d) >= {"signals", "class_base", "global_adjustment", "thresholds", "candidate_floor"}
    assert set(d["thresholds"]) == set(CLASS_NAMES)
    assert "Adaptive thresholds" in at.summary_line(sig)


def test_overrides_change_behaviour():
    """k_class=0 collapses the per-class spread to a single value."""
    at = AdaptiveThresholder(k_class=0.0)
    base = class_base_thresholds(at.config)
    assert len(set(round(v, 6) for v in base.values())) == 1
