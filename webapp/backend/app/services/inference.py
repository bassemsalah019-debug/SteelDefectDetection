"""
inference.py - the AI seam.

A thin, swappable, thread-safe wrapper around the project's existing ML code
(src/infer, src/adaptive_threshold, src/explain, src/report). The rest of the
backend depends only on this interface, so the model is mockable in tests and
replaceable (ONNX/TensorRT) without touching routes or services.
"""
from __future__ import annotations

import sys
import threading
from typing import Any

from PIL import Image as PILImage

from ..config import get_settings

settings = get_settings()
# Make the parent repo's `src` package importable (reuse all the ML as-is).
if str(settings.repo_root) not in sys.path:
    sys.path.insert(0, str(settings.repo_root))

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


class InferenceService:
    """Lazy singleton: loads the YOLO model + Eigen-CAM once, serves detections."""

    _instance: "InferenceService | None" = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._cam: Any = None
        self._run_lock = threading.Lock()  # model/CAM are not re-entrant-safe

    @classmethod
    def get(cls) -> "InferenceService":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO

            from src.explain import EigenCAM, enable_custom_modules

            enable_custom_modules()
            weights = settings.repo_root / settings.weights_path
            if not weights.exists():
                raise FileNotFoundError(f"Model weights not found: {weights}")
            self._model = YOLO(str(weights))
            if settings.enable_cam:
                self._cam = EigenCAM(self._model, device="cpu")
        return self._model

    def run(self, image: PILImage.Image, *, mode: str = "adaptive",
            conf: float = 0.25, imgsz: int = 640, want_cam: bool = True) -> dict:
        """Detect on one PIL image. Returns detections + annotated/CAM PIL images + signals."""
        from src.infer import predict, predict_adaptive

        with self._run_lock:
            model = self._ensure_loaded()
            signals = None
            if mode == "adaptive":
                ar = predict_adaptive(model, image, imgsz=imgsz, verbose=False)
                result, signals = ar.result, ar.detail["signals"]
            else:
                result = predict(model, image, conf=conf, imgsz=imgsz, verbose=False)

            boxes = result.boxes
            detections = []
            if boxes is not None:
                for b in boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    detections.append({
                        "cls_name": CLASS_NAMES[int(b.cls[0])],
                        "confidence": float(b.conf[0]),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

            annotated = PILImage.fromarray(result.plot()[:, :, ::-1])  # BGR->RGB
            cam_img = None
            if want_cam and self._cam is not None:
                from src.explain import overlay_cam

                heat = self._cam(image, imgsz=imgsz)
                cam_img = overlay_cam(image, heat)

        return {"detections": detections, "annotated": annotated, "cam": cam_img, "signals": signals}

    def report(self, detections: list[dict], *, lang: str = "en",
               image_meta: dict | None = None) -> dict:
        """Grounded bilingual report for a set of detections (LLM or KB fallback)."""
        from src.report import Detection, generate_report

        objs = [Detection(d["cls_name"], d["confidence"], [d["x1"], d["y1"], d["x2"], d["y2"]])
                for d in detections]
        rep = generate_report(objs, lang=lang, image_meta=image_meta)
        return {"text": rep["text"], "used_llm": rep["used_llm"]}
