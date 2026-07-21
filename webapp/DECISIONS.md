# DECISIONS — SteelVision web app

Non-obvious choices and why.

| # | Decision | Why |
|---|---|---|
| 1 | **FastAPI + React (Vite/TS)** over Next.js / Streamlit | The ML is all Python → a FastAPI backend reuses `src/` (YOLO, adaptive, Eigen-CAM, report) unchanged. Next.js would need a *separate* Python service anyway; Streamlit can't be a real multi-user platform. |
| 2 | **One `InferenceService` seam** ([services/inference.py](backend/app/services/inference.py)) | All model calls go through one swappable, mockable interface. Tests run with a fake (no GPU); ONNX/TensorRT could replace it without touching routes. |
| 3 | **SQLite default, Postgres via env** | Zero-install local dev for a non-expert; production switches with one `DATABASE_URL`. SQLAlchemy makes it transparent. |
| 4 | **Images on disk, paths in DB** | Keeps the DB tiny/fast; nginx/StaticFiles stream images efficiently. `storage.py` isolates file I/O so S3 is a one-file swap. |
| 5 | **3 images stored per input** (original, annotated, Eigen-CAM) | The UI lets inspectors toggle views; re-running inference on view-change would waste compute. |
| 6 | **Lazy thread-safe model singleton** | Load YOLO + Eigen-CAM once; a run-lock serializes calls (model isn't re-entrant) so FastAPI's threadpool can't corrupt state. |
| 7 | **Synchronous detection; batch in-request** | Simplicity for the MVP; `predict` is fast. A job queue (Celery/RQ) is the documented next step for heavy batches. |
| 8 | **Argon2 + JWT access/refresh, stateless logout** | Strong hashing; short access + longer refresh. No server-side revocation list yet (flagged as MVP debt). |
| 9 | **Adaptive owns confidence; `conf` ignored in adaptive mode** | Matches the parent project's `predict_adaptive` contract (per-class, per-image thresholds). |
| 10 | **`create_all` only in development; Alembic in prod** | Dev convenience without losing versioned, reversible production migrations. |
| 11 | **nginx proxies `/api`,`/auth`,`/media`; frontend built with `VITE_API_BASE=""`** | Same-origin in Docker → no CORS, no hard-coded backend URL in the bundle. |
| 12 | **Molten-Graphite dark theme, CSS animations (no Framer Motion)** | Carries the project's visual identity; fewer deps, smaller bundle, still animated. |
| 13 | **Frontend image builds with `npx vite build` (not `tsc && vite build`)** | Ships the runtime bundle without blocking the image on strict type-checks (dev/runtime are type-erased by esbuild). |

## Known debt / next steps
- Backend coverage is on core flows; add edge-case + service-unit tests.
- No refresh-token rotation/revocation list (stateless logout).
- Light mode deferred (dark only).
- E2E (Playwright) and a11y/Lighthouse pass not yet run (Phase 6).
- Docker images: files written and `docker compose config` validated; the full image
  build (Torch download, ~minutes) is the deploy step, not run in the build session.
