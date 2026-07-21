"""ORM models: User -> Inspection -> (Image -> Detection), Report."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), default="")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    inspections: Mapped[list["Inspection"]] = relationship(
        back_populates="user", cascade="all, delete-orphan")


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="Untitled inspection")
    mode: Mapped[str] = mapped_column(String(20), default="adaptive")  # fixed | adaptive
    conf: Mapped[float] = mapped_column(Float, default=0.25)
    imgsz: Mapped[int] = mapped_column(Integer, default=640)
    status: Mapped[str] = mapped_column(String(20), default="completed")  # pending|completed|failed
    n_images: Mapped[int] = mapped_column(Integer, default=0)
    n_defects: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)

    user: Mapped["User"] = relationship(back_populates="inspections")
    images: Mapped[list["Image"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan")
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan")
    reports: Mapped[list["Report"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    # web paths under the /media mount (e.g. "<inspection_id>/<id>.jpg")
    original_path: Mapped[str] = mapped_column(String(500))
    annotated_path: Mapped[str] = mapped_column(String(500), default="")
    cam_path: Mapped[str] = mapped_column(String(500), default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    n_defects: Mapped[int] = mapped_column(Integer, default=0)
    # adaptive signals (nullable -> only for adaptive mode)
    brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    inspection: Mapped["Inspection"] = relationship(back_populates="images")
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="image", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    cls_name: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    image: Mapped["Image"] = relationship(back_populates="detections")
    inspection: Mapped["Inspection"] = relationship(back_populates="detections")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    lang: Mapped[str] = mapped_column(String(5), default="en")  # en | ar
    text: Mapped[str] = mapped_column(Text, default="")
    used_llm: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    inspection: Mapped["Inspection"] = relationship(back_populates="reports")
