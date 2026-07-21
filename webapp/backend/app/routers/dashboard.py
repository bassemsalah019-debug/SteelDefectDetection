"""Dashboard: per-user aggregate statistics for the analytics view."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import Detection, Image, Inspection, User
from ..schemas import DashboardStats
from ..services.serializers import inspection_out

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def stats(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    uid = user.id

    total_inspections = db.query(func.count(Inspection.id)).filter(
        Inspection.user_id == uid).scalar() or 0
    total_images = db.query(func.count(Image.id)).join(
        Inspection, Image.inspection_id == Inspection.id).filter(
        Inspection.user_id == uid).scalar() or 0
    total_defects = db.query(func.count(Detection.id)).join(
        Inspection, Detection.inspection_id == Inspection.id).filter(
        Inspection.user_id == uid).scalar() or 0

    # defects per class
    class_counts: dict[str, int] = {}
    for name, n in (db.query(Detection.cls_name, func.count(Detection.id))
                    .join(Inspection, Detection.inspection_id == Inspection.id)
                    .filter(Inspection.user_id == uid)
                    .group_by(Detection.cls_name).all()):
        class_counts[name] = int(n)

    # mode split
    mode_split: dict[str, int] = {"fixed": 0, "adaptive": 0}
    for mode, n in (db.query(Inspection.mode, func.count(Inspection.id))
                    .filter(Inspection.user_id == uid)
                    .group_by(Inspection.mode).all()):
        mode_split[mode] = int(n)

    # last-14-days time series
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    insp_by_day: dict[str, int] = defaultdict(int)
    def_by_day: dict[str, int] = defaultdict(int)
    cutoff = datetime.combine(days[0], datetime.min.time())
    for insp in (db.query(Inspection)
                 .filter(Inspection.user_id == uid, Inspection.created_at >= cutoff).all()):
        key = insp.created_at.date().isoformat()
        insp_by_day[key] += 1
        def_by_day[key] += insp.n_defects
    over_time = [{"date": d.isoformat(), "inspections": insp_by_day[d.isoformat()],
                  "defects": def_by_day[d.isoformat()]} for d in days]

    recent = (db.query(Inspection).filter(Inspection.user_id == uid)
              .order_by(Inspection.created_at.desc()).limit(5).all())

    return {
        "total_inspections": int(total_inspections),
        "total_images": int(total_images),
        "total_defects": int(total_defects),
        "avg_defects_per_image": round(total_defects / total_images, 2) if total_images else 0.0,
        "class_counts": class_counts,
        "mode_split": mode_split,
        "over_time": over_time,
        "recent": [inspection_out(i) for i in recent],
    }
