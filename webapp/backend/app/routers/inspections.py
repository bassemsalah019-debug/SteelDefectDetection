"""Inspections: create (upload+detect), list, detail, delete, and report."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from ..database import get_db
from ..deps import get_current_user
from ..models import Detection, Image, Inspection, Report, User
from ..schemas import InspectionDetail, PageInspections, ReportOut, ReportRequest
from ..services.inference import InferenceService
from ..services.serializers import inspection_detail, inspection_out
from ..services.storage import delete_inspection_files, save_image

router = APIRouter(prefix="/api/inspections", tags=["inspections"])

_MAX_FILES = 20
_MAX_BYTES = 12 * 1024 * 1024  # 12 MB per image


def _load_owned(db: Session, insp_id: str, user: User) -> Inspection:
    insp = db.query(Inspection).options(
        selectinload(Inspection.images).selectinload(Image.detections),
        selectinload(Inspection.detections),
    ).filter(Inspection.id == insp_id, Inspection.user_id == user.id).first()
    if insp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inspection not found")
    return insp


@router.post("", response_model=InspectionDetail, status_code=status.HTTP_201_CREATED)
def create_inspection(
    files: list[UploadFile] = File(...),
    title: str = Form("Untitled inspection"),
    mode: str = Form("adaptive"),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    if mode not in ("fixed", "adaptive"):
        raise HTTPException(422, "mode must be 'fixed' or 'adaptive'")
    if imgsz not in (512, 640, 800):
        raise HTTPException(422, "imgsz must be 512, 640 or 800")
    if not files:
        raise HTTPException(422, "At least one image is required")
    if len(files) > _MAX_FILES:
        raise HTTPException(422, f"Too many images (max {_MAX_FILES})")

    insp = Inspection(user_id=user.id, title=title.strip() or "Untitled inspection",
                      mode=mode, conf=conf, imgsz=imgsz, status="completed")
    db.add(insp)
    db.flush()  # assign insp.id

    svc = InferenceService.get()
    total_defects = 0
    try:
        for uf in files:
            raw = uf.file.read()
            if len(raw) > _MAX_BYTES:
                raise HTTPException(413, f"{uf.filename}: image exceeds 12 MB")
            try:
                import io

                pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
            except (UnidentifiedImageError, OSError):
                raise HTTPException(422, f"{uf.filename}: not a valid image")

            out = svc.run(pil, mode=mode, conf=conf, imgsz=imgsz)
            img_id = uuid.uuid4().hex
            orig = save_image(pil, insp.id, f"{img_id}_orig.jpg")
            annotated = save_image(out["annotated"], insp.id, f"{img_id}_annotated.jpg")
            cam = save_image(out["cam"], insp.id, f"{img_id}_cam.jpg") if out["cam"] else ""

            sig = out["signals"] or {}
            image = Image(
                id=img_id, inspection_id=insp.id, filename=uf.filename or f"{img_id}.jpg",
                original_path=orig, annotated_path=annotated, cam_path=cam,
                width=pil.width, height=pil.height, n_defects=len(out["detections"]),
                brightness=sig.get("brightness"), quality=sig.get("quality"),
            )
            db.add(image)
            db.flush()
            for d in out["detections"]:
                db.add(Detection(image_id=image.id, inspection_id=insp.id, **d))
            total_defects += len(out["detections"])

        insp.n_images = len(files)
        insp.n_defects = total_defects
        db.commit()
    except HTTPException:
        db.rollback()
        delete_inspection_files(insp.id)
        raise
    except Exception as exc:  # detection failure -> mark failed, clean up
        db.rollback()
        delete_inspection_files(insp.id)
        raise HTTPException(500, f"Detection failed: {exc}") from exc

    db.refresh(insp)
    return inspection_detail(_load_owned(db, insp.id, user))


@router.get("", response_model=PageInspections)
def list_inspections(
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=100),
    mode: str | None = Query(None),
    cls: str | None = Query(None, description="filter to inspections containing this class"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    q = db.query(Inspection).filter(Inspection.user_id == user.id)
    if mode in ("fixed", "adaptive"):
        q = q.filter(Inspection.mode == mode)
    if cls:
        q = q.filter(Inspection.id.in_(
            db.query(Detection.inspection_id).filter(Detection.cls_name == cls)))
    total = q.count()
    rows = (q.order_by(Inspection.created_at.desc())
            .offset((page - 1) * page_size).limit(page_size).all())
    return {"items": [inspection_out(i) for i in rows], "total": total,
            "page": page, "page_size": page_size}


@router.get("/{insp_id}", response_model=InspectionDetail)
def get_inspection(insp_id: str, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)) -> dict:
    return inspection_detail(_load_owned(db, insp_id, user))


@router.delete("/{insp_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inspection(insp_id: str, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)) -> None:
    insp = db.query(Inspection).filter(
        Inspection.id == insp_id, Inspection.user_id == user.id).first()
    if insp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inspection not found")
    db.delete(insp)
    db.commit()
    delete_inspection_files(insp_id)


@router.post("/{insp_id}/report", response_model=ReportOut)
def generate_inspection_report(
    insp_id: str, body: ReportRequest,
    db: Session = Depends(get_db), user: User = Depends(get_current_user),
) -> Report:
    insp = _load_owned(db, insp_id, user)
    dets = [{"cls_name": d.cls_name, "confidence": d.confidence,
             "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2} for d in insp.detections]
    meta = {"inspection": insp.title, "n_images": insp.n_images,
            "mode": insp.mode, "imgsz": insp.imgsz}
    result = InferenceService.get().report(dets, lang=body.lang, image_meta=meta)
    report = Report(inspection_id=insp.id, lang=body.lang,
                    text=result["text"], used_llm=result["used_llm"])
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
