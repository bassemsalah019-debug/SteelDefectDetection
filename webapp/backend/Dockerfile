# Backend image. BUILD CONTEXT = repo root (it copies the reused `src/` + weights).
#   docker build -f webapp/backend/Dockerfile -t steelvision-api .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    ML_ROOT=/app \
    WEIGHTS_PATH=weights/best.pt \
    MPLCONFIGDIR=/tmp/mpl \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first (better layer caching)
COPY webapp/backend/requirements.txt webapp/backend/requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

# App + reused ML + production weights
COPY webapp/backend/ ./
COPY src/ ./src/
COPY results/baseline_640/weights/best.pt ./weights/best.pt

RUN useradd -m -u 1000 appuser && mkdir -p uploads /tmp/mpl /tmp/ultralytics \
    && chown -R appuser /app /tmp/mpl /tmp/ultralytics
USER appuser

EXPOSE 8000
# Apply migrations, then serve.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
