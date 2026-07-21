"""Application settings (12-factor: everything via env, with safe dev defaults)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PARENTS = Path(__file__).resolve().parents
BACKEND_DIR = _PARENTS[1]
# Dev layout: webapp/backend/app/config.py -> repo root is parents[3]. In a container the
# backend is copied to /app (config.py at /app/app/config.py, only 3 parents) -> fall back
# to /app. ML_ROOT env overrides this anyway, so the value only matters for local dev.
REPO_ROOT = _PARENTS[3] if len(_PARENTS) >= 4 else BACKEND_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SteelVision"
    environment: str = "development"

    # SQLite for zero-config dev; set DATABASE_URL to a Postgres URL in prod.
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'steelvision.db').as_posix()}"

    # Auth — OVERRIDE secret_key in production (.env).
    secret_key: str = "dev-secret-change-me-in-production-0123456789"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # CORS — the Vite dev server origins.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ML — root that contains the `src` package + weights. Defaults to the repo root
    # for local dev; in Docker set ML_ROOT=/app (where src/ and weights/ are copied).
    ml_root: str = str(REPO_ROOT)
    weights_path: str = "results/baseline_640/weights/best.pt"  # relative to ml_root
    upload_dir: str = (BACKEND_DIR / "uploads").as_posix()
    enable_cam: bool = True  # Eigen-CAM overlay (CPU); disable for max throughput.

    # Single-container deployments (e.g. HF Spaces): path to the built React app
    # (dist/). When set + present, FastAPI also serves the SPA same-origin.
    frontend_dist: str = ""

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def repo_root(self) -> Path:
        return Path(self.ml_root)


@lru_cache
def get_settings() -> Settings:
    return Settings()
