"""
explain.py - Explainable AI (XAI) for the steel-defect YOLO models.

Dependency-free Eigen-CAM for Ultralytics YOLOv8 / YOLO11 detectors, including
this project's custom Ghost + MPCA + SIoU model. No pip installs beyond what the
project already uses (torch, numpy, pillow, matplotlib).

Eigen-CAM (Muhammad & Yeasin, 2020) needs NO gradients: it takes the activation
map of a chosen layer and projects it onto its first principal component (SVD),
highlighting the image regions that dominate the detector's internal
representation - i.e. "where the model is looking". This is the established XAI
method for YOLO and is robust to the multi-instance, anchor-free detection head.

Usage
-----
    from src.explain import enable_custom_modules, load_yolo, EigenCAM, overlay_cam
    enable_custom_modules()                       # so improved/LZY best.pt unpickle
    model = load_yolo('results/improved_opt/weights/best.pt')
    cam   = EigenCAM(model, device='cpu')
    heat  = cam(pil_image, imgsz=640)             # HxW float map in [0, 1]
    vis   = overlay_cam(pil_image, heat)          # PIL.Image overlay
    cam.close()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.preprocessing import to_model_input


def enable_custom_modules(verbose: bool = False) -> None:
    """Register MPCA/SIoU and ResBlock_CBAM/WIoU so any project best.pt unpickles.

    Idempotent and inference-safe: the loss patches only matter during training,
    so calling this for a stock model is harmless. Stock models still load even
    if registration fails (e.g. src not importable).
    """
    try:
        from src.modules import register, register_lzy

        register(verbose=verbose)
        register_lzy(verbose=verbose)
    except Exception as exc:  # stock models still work without the custom layers
        if verbose:
            print("[explain] custom-module registration skipped:", exc)


def load_yolo(weights: str | Path):
    """Load an Ultralytics YOLO model, enabling custom modules first."""
    from ultralytics import YOLO

    enable_custom_modules()
    return YOLO(str(weights))


def _to_input(img: Image.Image, imgsz: int, device) -> torch.Tensor:
    # Canonical preprocessing: grayscale-replicate (what the detector was trained
    # on), then resize to imgsz - keeps the CAM input consistent with detection.
    gray3 = to_model_input(img)  # (H, W, 3) uint8, R == G == B
    arr = np.array(Image.fromarray(gray3).resize((imgsz, imgsz)))
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def _get_2d_projection(activation: np.ndarray) -> np.ndarray:
    """First-principal-component projection of a [C, H, W] activation -> [H, W]."""
    c, h, w = activation.shape
    reshaped = activation.reshape(c, h * w).T  # [H*W, C]
    reshaped = reshaped - reshaped.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(reshaped, full_matrices=False)
        proj = reshaped @ vt[0]  # project onto the dominant component -> [H*W]
    except np.linalg.LinAlgError:
        proj = reshaped.mean(axis=1)  # graceful fallback: mean activation
    proj = proj.reshape(h, w)
    if proj.mean() < 0:  # sign is arbitrary from SVD; make the response positive
        proj = -proj
    return proj


def _colormap(values: np.ndarray, name: str = "jet") -> np.ndarray:
    """Map a [H, W] array in [0, 1] to an [H, W, 3] uint8 RGB image (mpl-version safe)."""
    try:
        import matplotlib

        cmap = matplotlib.colormaps[name]  # matplotlib >= 3.7
    except Exception:  # older matplotlib
        import matplotlib.cm as cm

        cmap = cm.get_cmap(name)
    return (cmap(values)[:, :, :3] * 255).astype("uint8")


class EigenCAM:
    """Eigen-CAM for an Ultralytics YOLO detector (no gradients required).

    Args:
        model: an ``ultralytics.YOLO`` instance.
        layer_index: index into the model's layer Sequential to explain. Default
            is the last layer before the Detect head - a single, semantic feature
            map that works for stock YOLOv8/YOLO11 and the improved model alike.
        device: 'cpu' (default for app/CI) or 'cuda'.
    """

    def __init__(self, model, layer_index: int | None = None, device: str | None = None):
        self.model = model
        self.net = model.model  # the DetectionModel (nn.Module)
        # IMPORTANT: do NOT move the model here. The model is usually shared with the
        # detection path (and Streamlit caches it as one object); forcing it onto a
        # fixed device would desync it from model.predict()'s auto-selected device and
        # raise "Input type (cuda) and weight type (cpu) should be the same". We follow
        # the model's *current* device at call time instead (see _device).
        self.net.eval()
        self._fallback_device = device  # only used if the model has no parameters
        layers = self.net.model  # nn.Sequential of the YAML-defined layers
        self.layer_index = layer_index if layer_index is not None else len(layers) - 2
        self._act: dict[str, torch.Tensor] = {}
        self._handle = layers[self.layer_index].register_forward_hook(
            lambda m, i, o: self._act.__setitem__("a", o)
        )

    def _device(self) -> torch.device:
        """The model's current device - followed dynamically so CAM never conflicts
        with model.predict()'s device."""
        try:
            return next(self.net.parameters()).device
        except StopIteration:
            return torch.device(self._fallback_device or "cpu")

    @torch.no_grad()
    def __call__(self, img: Image.Image, imgsz: int = 640) -> np.ndarray:
        """Return an [imgsz, imgsz] heatmap in [0, 1] for one PIL image."""
        self.net(_to_input(img, imgsz, self._device()))
        a = self._act["a"]
        if isinstance(a, (list, tuple)):
            a = a[0]
        a = a[0].detach().cpu().float().numpy()  # [C, H, W]
        cam = _get_2d_projection(a)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((imgsz, imgsz), Image.BILINEAR)
        return np.asarray(cam_img, dtype=np.float32) / 255.0

    def close(self) -> None:
        """Remove the forward hook. Safe to call more than once."""
        if getattr(self, "_handle", None) is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def overlay_cam(img: Image.Image, heat: np.ndarray, alpha: float = 0.5, cmap: str = "jet") -> Image.Image:
    """Blend a [H, W] heatmap in [0, 1] over the image with a colormap -> PIL.Image."""
    size = (heat.shape[1], heat.shape[0])
    base = np.asarray(img.convert("RGB").resize(size), dtype=np.float32)
    colored = _colormap(heat, cmap).astype(np.float32)
    blend = (base * (1 - alpha) + colored * alpha).clip(0, 255).astype("uint8")
    return Image.fromarray(blend)


def attention_summary(heat: np.ndarray, detections=None, img_size=None,
                      thresh: float = 0.5) -> str:
    """One-line structured summary of WHERE the Eigen-CAM looks - feeds the report.

    Args:
        heat: [H, W] saliency map in [0, 1].
        detections: optional iterable of objects with a ``.bbox`` (xyxy) or
            (cls, conf, bbox) tuples, used to count detections inside the hot region.
        img_size: (W, H) of the image the bboxes are in (needed to map them onto heat).
        thresh: activation level that counts as "high attention".
    """
    heat = np.asarray(heat, dtype=np.float32)
    h, w = heat.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    total = float(heat.sum()) + 1e-8
    cy = float((heat * ys).sum()) / total / max(h - 1, 1)
    cx = float((heat * xs).sum()) / total / max(w - 1, 1)
    vrow = "top" if cy < 0.34 else "bottom" if cy > 0.66 else "middle"
    hcol = "left" if cx < 0.34 else "right" if cx > 0.66 else "center"
    region = "center" if (vrow == "middle" and hcol == "center") else f"{vrow}-{hcol}"
    frac = float((heat >= thresh).mean())
    py, px = np.unravel_index(int(np.argmax(heat)), heat.shape[:2])
    parts = [
        f"Eigen-CAM attention concentrates in the {region} region "
        f"(peak near x={px / max(w - 1, 1):.0%}, y={py / max(h - 1, 1):.0%}); "
        f"{frac:.0%} of the saliency map exceeds {thresh:.1f} activation."
    ]
    if detections and img_size:
        W, H = img_size
        hot = heat >= thresh
        aligned = 0
        dets = list(detections)
        for d in dets:
            box = getattr(d, "bbox", None)
            if box is None and isinstance(d, (list, tuple)) and len(d) >= 3:
                box = d[2]
            if not box:
                continue
            x1, y1, x2, y2 = box
            gx = min(int(((x1 + x2) / 2) / max(W, 1) * w), w - 1)
            gy = min(int(((y1 + y2) / 2) / max(H, 1) * h), h - 1)
            aligned += int(bool(hot[gy, gx]))
        parts.append(f"{aligned} of {len(dets)} detections fall inside the high-attention area.")
    return " ".join(parts)
