"""
Preprocessing parity test - the HARD CONSTRAINT that the app path and the
inference/eval path preprocess identically, and that the RGB->grayscale bug is
fixed at the source.

Light by design: imports only src.preprocessing + src.infer (numpy + PIL), no
torch/ultralytics/streamlit, so it runs in CI on a fresh checkout.

    python -m pytest tests/test_preprocessing_parity.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infer import preprocess  # the app/inference entry point
from src.preprocessing import channels_equal, to_model_input  # the canonical fn

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _sample_color_array(seed: int = 0) -> np.ndarray:
    """A genuinely colourful (R != G != B) image to expose the bug if present."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_app_path_equals_inference_path_pil():
    """app path (src.infer.preprocess) == canonical (to_model_input), byte-identical."""
    img = Image.fromarray(_sample_color_array(1))
    assert np.array_equal(preprocess(img), to_model_input(img))


def test_app_path_equals_inference_path_array():
    arr = _sample_color_array(2)
    assert np.array_equal(preprocess(arr), to_model_input(arr))


def test_output_is_grayscale_replicated():
    """The fix: output must be 3-channel with R == G == B (what the model trained on)."""
    out = to_model_input(_sample_color_array(3))
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    assert channels_equal(out), "channels not equal -> colour leaked into model input (bug)"


def test_color_input_actually_collapses():
    """A colour image must NOT survive as colour - guards against regressing the bug."""
    color = _sample_color_array(4)
    # the raw colour image has distinct channels...
    assert not channels_equal(color)
    # ...but after preprocessing the channels are identical.
    assert channels_equal(to_model_input(color))


def test_pil_and_array_agree():
    """Same pixels via PIL(RGB) or numpy(RGB) preprocess identically."""
    arr = _sample_color_array(5)
    assert np.array_equal(to_model_input(arr), to_model_input(Image.fromarray(arr)))


def test_bgr_flag_matches_channel_swap():
    """is_bgr=True on a BGR array == passing the RGB-flipped array as RGB."""
    rgb = _sample_color_array(6)
    bgr = rgb[:, :, ::-1].copy()
    assert np.array_equal(to_model_input(bgr, is_bgr=True), to_model_input(rgb))


def test_idempotent_on_gray():
    """Re-preprocessing an already-gray (NEU-DET-like) input is a no-op."""
    once = to_model_input(_sample_color_array(7))
    assert np.array_equal(once, to_model_input(once))


def test_matches_pil_convert_L():
    """Luma must match PIL's convert('L') so PIL and numpy paths never diverge."""
    arr = _sample_color_array(8)
    pil_gray = np.array(Image.fromarray(arr).convert("L"))
    assert np.array_equal(to_model_input(arr)[..., 0], pil_gray)
