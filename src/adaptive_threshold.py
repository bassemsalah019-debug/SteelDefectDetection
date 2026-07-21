"""
adaptive_threshold.py - Adaptive Real-Time Confidence Thresholding for the
steel-defect YOLOv8n detector (industrial deployment).

Why this exists
---------------
A single fixed confidence threshold (the app's default 0.25) is a poor operating
point on NEU-DET because the six classes are wildly uneven and the imaging
conditions on a real strip-mill line drift (lighting, focus, scale). At a fixed
0.25, easy classes (`patches` AP 0.92) emit low-value false positives while hard,
low-contrast classes (`crazing` AP 0.44) get their genuine-but-timid detections
truncated -> missed defects. The fixed threshold cannot be right for both.

This module computes a *per-class, per-image* confidence threshold from four cheap
signals, applied as a POST-PROCESSING filter on the detector's output:

  1. class difficulty   - a per-class anchor from the measured baseline AP@0.5.
                          Hard classes get a lower anchor (protect recall); easy
                          classes a higher anchor (suppress false positives).
  2. image brightness   - mean luma; under/over-exposed frames depress confidence,
                          so the threshold is lowered to preserve recall.
  3. detection density  - candidate count; crowded frames raise the threshold
                          (precision / fewer nuisance alerts), sparse frames lower
                          it (don't miss the rare defect).
  4. image quality      - contrast + sharpness (Laplacian focus); blurry / flat
                          frames lower the threshold to compensate for lost signal.

Design constraints honoured
---------------------------
- PURE POST-PROCESSING. The model graph is untouched, so ONNX/TensorRT export is
  unaffected and this works identically on any backend (PyTorch / ORT / TRT).
- TORCH-FREE and dependency-light (numpy; cv2 optional with a numpy fallback) so
  it imports cheaply and is unit-testable without a GPU - same ethos as
  src/explain.py and src/preprocessing.py.
- O(H*W) signals computed on a 256-px grayscale thumbnail: ~0.1-0.3 ms, negligible
  against the ~6 ms network forward pass -> real-time on edge devices is preserved.
- Signals use the grayscale luma, consistent with the canonical grayscale model
  input (src/preprocessing.to_model_input).

All weights/anchors live in `AdaptiveConfig` with documented defaults; nothing is
hidden. Recalibrate `class_ap` from a fresh `src/eval.py` run when the model changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

# Measured TEST AP@0.5 of the PRODUCTION model (results/baseline_640/results.json,
# 2026-06-13). This is the class-difficulty prior. Update it if the model is retrained.
BASELINE_AP50: dict[str, float] = {
    "crazing": 0.4403,
    "inclusion": 0.8581,
    "patches": 0.9247,
    "pitted_surface": 0.8624,
    "rolled-in_scale": 0.6356,
    "scratches": 0.7938,
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdaptiveConfig:
    """All tunables for adaptive thresholding. Defaults are deliberately gentle so
    the adaptive operating point stays close to the fixed 0.25 baseline."""

    # Per-class anchor: base_c = t0 + k_class * (AP_c - mean_AP)
    t0: float = 0.25                       # nominal centre (matches the app default)
    k_class: float = 0.30                  # how strongly class difficulty shifts the anchor
    class_ap: Mapping[str, float] = field(default_factory=lambda: dict(BASELINE_AP50))

    # Global image-level adjustments (added to every class anchor)
    w_bright: float = 0.08                 # max downward shift for bad exposure
    w_quality: float = 0.10                # max downward shift for blur / low contrast
    w_density: float = 0.06                # max +/- shift from detection density

    # Signal reference points (calibratable; see report for derivation)
    bright_ideal: float = 0.50             # ideal mean luma (0-1)
    bright_half: float = 0.50              # luma distance that maps to full penalty
    contrast_ref: float = 0.18             # std/255 that counts as "good contrast"
    sharp_ref: float = 80.0                # Laplacian-variance scale for "in focus"
    density_ref: float = 3.0               # candidate count treated as neutral
    density_scale: float = 3.0             # soft width of the density response

    # Hard bounds + candidate gathering floor
    t_min: float = 0.08                    # never go below this (avoid noise flood)
    t_max: float = 0.60                    # never go above this (avoid silent misses)
    candidate_floor: float = 0.05          # conf used to gather candidates before filtering
    signal_resize: int = 256               # thumbnail size for O(1)-ish signal cost

    def mean_ap(self) -> float:
        vals = list(self.class_ap.values())
        return float(sum(vals) / len(vals)) if vals else 0.75


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImageSignals:
    """Cheap, interpretable per-image signals driving the adaptive threshold."""
    brightness: float          # mean luma, 0-1
    contrast: float            # std luma / 255, 0-~0.3
    sharpness: float           # variance of Laplacian (0-255 scale)
    quality: float             # combined contrast+sharpness quality, 0-1
    density: int               # number of candidate detections (>= candidate_floor)

    def as_dict(self) -> dict:
        return {
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "sharpness": round(self.sharpness, 2),
            "quality": round(self.quality, 4),
            "density": int(self.density),
        }


def _to_luma(img: np.ndarray) -> np.ndarray:
    """(H,W) or (H,W,3) array -> (H,W) float32 luma in [0,255].

    The canonical model input is grayscale-replicated (R==G==B), so channel 0 IS
    the luma; for a genuine colour array we fall back to ITU-R 601 luma.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        ch = arr[..., :3].astype(np.float32)
        if np.array_equal(ch[..., 0], ch[..., 1]) and np.array_equal(ch[..., 1], ch[..., 2]):
            return ch[..., 0]
        return 0.299 * ch[..., 0] + 0.587 * ch[..., 1] + 0.114 * ch[..., 2]
    raise ValueError(f"Unsupported image shape for signals: {arr.shape}")


def _laplacian_var(g: np.ndarray) -> float:
    """Variance of the Laplacian = focus / sharpness measure. cv2 if present, else
    a 4-neighbour numpy stencil (identical idea, no dependency)."""
    try:
        import cv2

        return float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var())
    except Exception:
        lap = (
            -4.0 * g
            + np.roll(g, 1, 0) + np.roll(g, -1, 0)
            + np.roll(g, 1, 1) + np.roll(g, -1, 1)
        )
        return float(lap[1:-1, 1:-1].var())


def _resize_thumb(g: np.ndarray, size: int) -> np.ndarray:
    """Cheap nearest-ish downscale to `size` on the long side (numpy stride slice).
    Keeps signal cost ~constant regardless of input resolution."""
    h, w = g.shape[:2]
    m = max(h, w)
    if m <= size:
        return g
    step = int(np.ceil(m / size))
    return g[::step, ::step]


def compute_image_signals(img: np.ndarray, *, density: int = 0,
                          config: AdaptiveConfig | None = None) -> ImageSignals:
    """Compute brightness / contrast / sharpness / quality from an image (+ density).

    Args:
        img: model input or raw frame; (H,W) gray or (H,W,3). Grayscale-replicated
            inputs are used as-is (channel 0).
        density: number of candidate detections for this image (>= candidate_floor).
        config: AdaptiveConfig (for the signal resize + quality references).
    """
    cfg = config or AdaptiveConfig()
    g = _resize_thumb(_to_luma(img), cfg.signal_resize)

    brightness = float(g.mean()) / 255.0
    contrast = float(g.std()) / 255.0
    sharpness = _laplacian_var(g)

    contrast_q = float(np.clip(contrast / max(cfg.contrast_ref, 1e-6), 0.0, 1.0))
    sharp_q = float(sharpness / (sharpness + cfg.sharp_ref)) if cfg.sharp_ref > 0 else 1.0
    quality = 0.5 * contrast_q + 0.5 * sharp_q

    return ImageSignals(brightness=brightness, contrast=contrast, sharpness=sharpness,
                        quality=quality, density=int(density))


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
def class_base_thresholds(config: AdaptiveConfig | None = None) -> dict[str, float]:
    """Per-class anchor from class difficulty: base_c = t0 + k_class*(AP_c - mean_AP).

    Hard (low-AP) classes -> lower anchor (protect recall); easy classes -> higher."""
    cfg = config or AdaptiveConfig()
    mean_ap = cfg.mean_ap()
    out = {}
    for name in CLASS_NAMES:
        ap = float(cfg.class_ap.get(name, mean_ap))
        out[name] = float(np.clip(cfg.t0 + cfg.k_class * (ap - mean_ap), cfg.t_min, cfg.t_max))
    return out


def global_adjustment(signals: ImageSignals, config: AdaptiveConfig | None = None) -> dict:
    """The three image-level deltas (same for every class), returned itemised so the
    app / report can show *why* the threshold moved."""
    cfg = config or AdaptiveConfig()

    dev_b = float(np.clip(abs(signals.brightness - cfg.bright_ideal) / max(cfg.bright_half, 1e-6), 0.0, 1.0))
    d_bright = -cfg.w_bright * dev_b

    d_quality = -cfg.w_quality * (1.0 - float(np.clip(signals.quality, 0.0, 1.0)))

    d_density = cfg.w_density * float(np.tanh((signals.density - cfg.density_ref) / max(cfg.density_scale, 1e-6)))

    return {
        "brightness": round(d_bright, 4),
        "quality": round(d_quality, 4),
        "density": round(d_density, 4),
        "total": round(d_bright + d_quality + d_density, 4),
    }


def adaptive_thresholds(signals: ImageSignals,
                        config: AdaptiveConfig | None = None) -> dict[str, float]:
    """Final per-class thresholds for the current image:
        T_c = clip(base_c + d_bright + d_quality + d_density, t_min, t_max)."""
    cfg = config or AdaptiveConfig()
    base = class_base_thresholds(cfg)
    delta = global_adjustment(signals, cfg)["total"]
    return {name: float(np.clip(base[name] + delta, cfg.t_min, cfg.t_max)) for name in CLASS_NAMES}


def thresholds_by_id(signals: ImageSignals,
                     config: AdaptiveConfig | None = None) -> dict[int, float]:
    """Same as adaptive_thresholds but keyed by class id (for filtering YOLO output)."""
    by_name = adaptive_thresholds(signals, config)
    return {i: by_name[name] for i, name in enumerate(CLASS_NAMES)}


def keep_mask(cls_ids: np.ndarray, confs: np.ndarray,
              thr_by_id: Mapping[int, float]) -> np.ndarray:
    """Boolean mask of detections to KEEP: conf >= adaptive threshold for its class."""
    cls_ids = np.asarray(cls_ids).astype(int).ravel()
    confs = np.asarray(confs).astype(float).ravel()
    if cls_ids.size == 0:
        return np.zeros(0, dtype=bool)
    thr = np.array([thr_by_id.get(int(c), 1.0) for c in cls_ids], dtype=float)
    return confs >= thr


# --------------------------------------------------------------------------- #
# Convenience facade (used by the app + inference wrapper)
# --------------------------------------------------------------------------- #
class AdaptiveThresholder:
    """Bundles config + the compute->threshold->explain flow for one image.

    Typical use (after running predict at config.candidate_floor):
        at = AdaptiveThresholder()
        sig = at.signals(model_input, density=len(candidate_boxes))
        thr = at.thresholds_by_id(sig)
        mask = keep_mask(cls_ids, confs, thr)
    """

    def __init__(self, config: AdaptiveConfig | None = None, **overrides):
        cfg = config or AdaptiveConfig()
        self.config = replace(cfg, **overrides) if overrides else cfg

    def signals(self, img: np.ndarray, *, density: int = 0) -> ImageSignals:
        return compute_image_signals(img, density=density, config=self.config)

    def thresholds(self, signals: ImageSignals) -> dict[str, float]:
        return adaptive_thresholds(signals, self.config)

    def thresholds_by_id(self, signals: ImageSignals) -> dict[int, float]:
        return thresholds_by_id(signals, self.config)

    def describe(self, signals: ImageSignals) -> dict:
        """Full, displayable breakdown for the Streamlit UI and the LLM report."""
        return {
            "signals": signals.as_dict(),
            "class_base": {k: round(v, 4) for k, v in class_base_thresholds(self.config).items()},
            "global_adjustment": global_adjustment(signals, self.config),
            "thresholds": {k: round(v, 4) for k, v in self.thresholds(signals).items()},
            "candidate_floor": self.config.candidate_floor,
        }

    def summary_line(self, signals: ImageSignals) -> str:
        """One human sentence (feeds the bilingual report's XAI/context section)."""
        thr = self.thresholds(signals)
        lo = min(thr, key=thr.get)
        hi = max(thr, key=thr.get)
        return (
            f"Adaptive thresholds: brightness {signals.brightness:.2f}, quality "
            f"{signals.quality:.2f}, {signals.density} candidate(s). Per-class conf "
            f"ranges {thr[lo]:.2f} ({lo}) to {thr[hi]:.2f} ({hi})."
        )
