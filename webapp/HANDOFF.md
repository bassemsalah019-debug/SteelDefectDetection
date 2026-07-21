# SteelVision — Handoff Report

Full-stack steel-surface **defect inspection platform** built on top of the existing
YOLOv8n research project. Status: **MVP complete and running**.

## 1. What was built

**Backend (FastAPI)** — `webapp/backend/`
- JWT auth (signup / login / refresh / logout / me), argon2 password hashing.
- Inspections: upload 1–20 images → detect (Fixed/Adaptive) → store; list (paginated +
  filter), detail, delete; per-inspection EN/AR report.
- Dashboard aggregates: totals, defects-per-class, mode split, 14-day trend, recent.
- `InferenceService` seam reusing the project's ML (`src/infer`, `src/adaptive_threshold`,
  `src/explain`, `src/report`).
- SQLAlchemy models + **Alembic** migrations; SQLite (dev) / Postgres (prod).
- Static `/media` image serving; one consistent JSON error shape; auto OpenAPI.

**Frontend (React + Vite + TypeScript)** — `webapp/frontend/`
- Pages: **Login/Signup, Dashboard, New Inspection (drag-drop upload), History, Inspection
  Detail** (toggle Detections / Eigen-CAM / Original, per-image tables, adaptive signals,
  Generate report).
- "Molten-Graphite" dark design system, CSS animations, KPI tiles, glowing class bars,
  mini activity chart; **bilingual EN/AR with RTL**; responsive; loading/empty/error states.
- Typed API client, auth context (token persistence + guarded routes), i18n context.

**Ops** — Dockerfiles (backend + nginx frontend), `docker-compose.yml` (Postgres + backend
+ frontend), `.dockerignore`, one-click `.bat` launchers.

## 2. Architecture
```
React (5173 / nginx 8080)  ──HTTP/JSON──▶  FastAPI (8000)
  TanStack-free fetch client                routes → services → data
  auth + i18n contexts                      InferenceService → src/ ML (YOLO+CAM+report)
  Molten-Graphite UI                        SQLAlchemy + Alembic → SQLite/Postgres
                                            /media static image files
```
Layering: `routes → services → data`; ML isolated behind one swappable interface.

## 3. How to run
- **Local (easiest):** double-click `webapp/start-app.bat` → http://localhost:5173
- **Local (manual):** backend `uvicorn app.main:app --port 8000`; frontend `npm run dev`.
- **Docker:** `cd webapp && docker compose up --build` → http://localhost:8080
- **Test account:** `inspector@steel.io` / `password123`
- Full details: `webapp/README.md`.

## 4. Verification (run this session)
| Check | Result |
|---|---|
| Backend health/auth/inspections/dashboard/report (real HTTP) | ✅ all 200, correct payloads |
| End-to-end: upload real steel image → 8 defects detected, signals, /media image served | ✅ |
| Frontend production build | ✅ 30 modules, 0 errors |
| Frontend dev server + both ports live | ✅ |
| Backend unit tests | ✅ **15/15 passing** |
| Alembic: autogenerate + `upgrade head` builds all 5 tables | ✅ |
| `docker compose config` | ✅ valid |

## 5. Limitations / known debt
- Refresh-token rotation/revocation not implemented (stateless logout).
- Light mode deferred; only the dark theme ships.
- E2E (Playwright) + Lighthouse a11y/perf pass not yet done (Phase 6).
- Detection is synchronous; very large batches would benefit from a job queue.
- Docker image build (Torch, ~minutes) is set up + config-validated but not built in-session.

## 6. Prioritized next steps
1. Build/run the Docker stack once and smoke-test the deployed URLs.
2. Playwright E2E for the critical path (signup → upload → report) + Lighthouse pass.
3. Refresh-token rotation + token revocation on logout.
4. Background job queue for large batch inspections.
5. Light theme + a11y polish (focus states already present; audit contrast/ARIA).
6. S3-backed storage (swap `services/storage.py`).

## 7. 60-second demo script
1. Open **http://localhost:5173**, log in (`inspector@steel.io` / `password123`).
2. **Dashboard** shows KPIs + the sample inspection (8 crazing defects).
3. **＋ New Inspection** → drop `data/neu-det-yolo/images/test/scratches_1.jpg` →
   **Adaptive** → **Run inspection**.
4. On the detail page, toggle **Detections / Eigen-CAM / Original**; read the defect table
   and adaptive signals.
5. Switch sidebar language to **ع** (RTL), click **Generate report** → Arabic report.
6. Open **History** → see both inspections; open **/docs** to show the typed API.
