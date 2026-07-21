"""Health + service info endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/api/health")
def api_health() -> dict:
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}
