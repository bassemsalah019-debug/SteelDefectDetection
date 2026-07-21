"""Pydantic v2 request/response schemas (the API contract)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

_ORM = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=120)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = _ORM
    id: str
    email: EmailStr
    full_name: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


class RefreshIn(BaseModel):
    refresh_token: str


# --------------------------------------------------------------------------- #
# Detections / images / inspections
# --------------------------------------------------------------------------- #
class DetectionOut(BaseModel):
    model_config = _ORM
    id: str
    cls_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ImageOut(BaseModel):
    model_config = _ORM
    id: str
    filename: str
    original_url: str = ""
    annotated_url: str = ""
    cam_url: str = ""
    width: int
    height: int
    n_defects: int
    brightness: Optional[float] = None
    quality: Optional[float] = None
    detections: list[DetectionOut] = []


class InspectionOut(BaseModel):
    """List/summary view."""
    model_config = _ORM
    id: str
    title: str
    mode: str
    conf: float
    imgsz: int
    status: str
    n_images: int
    n_defects: int
    created_at: datetime


class InspectionDetail(InspectionOut):
    """Full view with images + detections + class counts."""
    images: list[ImageOut] = []
    class_counts: dict[str, int] = {}


class ReportOut(BaseModel):
    model_config = _ORM
    id: str
    lang: str
    text: str
    used_llm: bool
    created_at: datetime


class ReportRequest(BaseModel):
    lang: Literal["en", "ar"] = "en"


class PageInspections(BaseModel):
    items: list[InspectionOut]
    total: int
    page: int
    page_size: int


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class TimePoint(BaseModel):
    date: str
    inspections: int
    defects: int


class DashboardStats(BaseModel):
    total_inspections: int
    total_images: int
    total_defects: int
    avg_defects_per_image: float
    class_counts: dict[str, int]
    mode_split: dict[str, int]
    over_time: list[TimePoint]
    recent: list[InspectionOut]
