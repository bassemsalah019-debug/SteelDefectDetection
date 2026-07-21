"""Test fixtures: isolated temp DB + uploads, and a MOCKED AI seam (no model needed)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services import storage  # noqa: E402
from app.services.inference import InferenceService  # noqa: E402


class _FakeInference:
    """Deterministic stand-in for the real YOLO/Eigen-CAM service."""

    def run(self, image, *, mode="adaptive", conf=0.25, imgsz=640, want_cam=True):
        return {
            "detections": [{"cls_name": "crazing", "confidence": 0.91,
                            "x1": 1.0, "y1": 1.0, "x2": 9.0, "y2": 9.0}],
            "annotated": PILImage.new("RGB", (16, 16), "gray"),
            "cam": PILImage.new("RGB", (16, 16), "gray") if want_cam else None,
            "signals": {"brightness": 0.5, "quality": 0.8, "sharpness": 120.0, "density": 1},
        }

    def report(self, detections, *, lang="en", image_meta=None):
        return {"text": "## Executive Summary\nDetected 1 defect (crazing).", "used_llm": False}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                           connect_args={"check_same_thread": False})
    Testing = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = Testing()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(storage, "_ROOT", uploads)
    monkeypatch.setattr(InferenceService, "get", staticmethod(lambda: _FakeInference()))

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def img_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (32, 32), "gray").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def auth_client(client):
    r = client.post("/auth/signup",
                    json={"email": "tester@steel.io", "password": "password123", "full_name": "Tester"})
    assert r.status_code == 201, r.text
    client.headers.update({"Authorization": f"Bearer {r.json()['access_token']}"})
    return client
