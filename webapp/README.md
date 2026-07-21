# SteelVision — full-stack steel-defect inspection platform

A login-protected web app where an inspector uploads steel-surface images (single or
batch), the system **detects defects** (Fixed/Adaptive thresholds), shows **Eigen-CAM**
heatmaps, generates a grounded **bilingual EN/AR report**, stores every inspection, and
charts defect trends on a **dashboard**.

**Stack:** FastAPI + SQLAlchemy + (SQLite dev / Postgres prod) · React + Vite + TypeScript
(Molten-Graphite UI, Framer-free CSS animations, EN/AR + RTL). The ML (YOLOv8n + adaptive
thresholding + Eigen-CAM + report) is the parent project's `src/`, reused behind one
`InferenceService` seam.

---

## ▶️ Run it locally (easiest — Windows)

Double-click **`webapp\start-app.bat`** — it starts the backend + frontend in two windows
and opens your browser. Then go to **http://localhost:5173**.

### Or manually (two terminals, from the project root)
```powershell
# Terminal 1 — backend
& "C:\Users\student\Downloads\files\.venv\Scripts\Activate.ps1"
cd webapp\backend
python -m uvicorn app.main:app --port 8000

# Terminal 2 — frontend
cd webapp\frontend
npm install        # first time only
npm run dev
```

| URL | What |
|---|---|
| **http://localhost:5173** | **The app (frontend)** |
| http://localhost:8000 | Backend API |
| http://localhost:8000/docs | Interactive API docs (Swagger) |

**Test account:** `inspector@steel.io` / `password123` (or click **Sign up**).
No `.env` needed — safe dev defaults (SQLite + dev secret).

---

## 🐳 Run with Docker (production-style: Postgres + nginx)
```bash
cd webapp
docker compose up --build
```
Then open **http://localhost:8080** (frontend; API at http://localhost:8000, docs at
`/docs`). The first build downloads Torch/Ultralytics (a few minutes, ~2 GB image).
Compose runs **Postgres**, applies **Alembic migrations**, and serves the React app via
**nginx** (which proxies `/api`, `/auth`, `/media` to the backend — no CORS issues).

---

## 🧪 Tests
```powershell
cd webapp\backend
python -m pytest tests -q        # 15 tests: auth, inspections, dashboard (AI seam mocked)
```

## 🗄️ Database
- **Dev:** one SQLite file `webapp/backend/steelvision.db`. **Images** live on disk under
  `webapp/backend/uploads/<inspection_id>/...` (original + annotated + Eigen-CAM), served at
  `/media/...`. The DB stores paths, not bytes.
- **Migrations (Alembic):**
  ```bash
  cd webapp/backend
  alembic upgrade head                          # apply
  alembic revision --autogenerate -m "change"   # after editing models.py
  ```
- **Postgres:** set `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/steelvision`.

---

## 📁 Structure
```
webapp/
├── start-app.bat / start-backend.bat / start-frontend.bat   # one-click launchers
├── docker-compose.yml
├── backend/                 # FastAPI
│   ├── app/
│   │   ├── main.py · config.py · database.py · models.py · schemas.py
│   │   ├── security.py · deps.py
│   │   ├── routers/  auth · inspections · dashboard · health
│   │   └── services/ inference (AI seam) · storage · serializers
│   ├── alembic/             # migrations
│   ├── tests/               # pytest
│   ├── Dockerfile · requirements*.txt · .env.example
└── frontend/                # React + Vite + TS
    ├── src/
    │   ├── api.ts · auth.tsx · i18n.tsx · theme.css · config.ts
    │   ├── components.tsx · App.tsx · main.tsx
    │   └── pages/  Login · Dashboard · NewInspection · History · Detail
    └── Dockerfile · nginx.conf
```

See **`DECISIONS.md`** for design choices and **`HANDOFF.md`** for the full project report.
