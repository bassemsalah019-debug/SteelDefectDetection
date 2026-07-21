"""SteelVision API — app factory, middleware, error shape, routers, static media."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings
from .database import Base, engine
from . import models  # noqa: F401  (register tables before create_all)
from .routers import auth, dashboard, health, inspections

settings = get_settings()

# Dev convenience: create tables on boot. In production set ENVIRONMENT=production
# and let Alembic own the schema (`alembic upgrade head`).
if settings.environment == "development":
    Base.metadata.create_all(bind=engine)

# Ensure the uploads dir exists, then serve it read-only at /media.
upload_dir = Path(settings.upload_dir)
upload_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=f"{settings.app_name} API",
    version="0.1.0",
    description="Steel surface defect inspection platform — detection, XAI, bilingual reports.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- one consistent error shape: {"error": {"code", "message", "details?"}} ----
@app.exception_handler(StarletteHTTPException)
async def _http_exc(_, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
async def _validation_exc(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": "Validation error",
                           "details": jsonable_encoder(exc.errors())}},
    )


@app.on_event("startup")
def _warmup_model() -> None:
    """Load + warm the YOLO model in a background thread so the FIRST user detection
    isn't slowed by a cold model load (which can exceed a hosting proxy's timeout)."""
    import threading

    def _run() -> None:
        try:
            from PIL import Image

            from .services.inference import InferenceService

            InferenceService.get().run(Image.new("RGB", (200, 200), "gray"),
                                       mode="fixed", imgsz=640, want_cam=False)
        except Exception:
            pass  # best-effort; real requests still load lazily

    threading.Thread(target=_run, daemon=True).start()


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(inspections.router)
app.include_router(dashboard.router)

app.mount("/media", StaticFiles(directory=str(upload_dir)), name="media")

# Single-container deployments (HF Spaces): serve the built React SPA same-origin.
# Registered LAST so the API routers + /media take precedence; the catch-all returns
# index.html for client-side routes (e.g. /history on refresh).
if settings.frontend_dist:
    from fastapi.responses import FileResponse

    _dist = Path(settings.frontend_dist)
    if _dist.exists():
        _assets = _dist / "assets"
        if _assets.exists():
            app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def _spa(full_path: str):
            candidate = _dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(_dist / "index.html"))
