# Hugging Face Space (Docker SDK) running the Streamlit Detection Studio on CPU.
# HF Docker Spaces serve on port 7860.
FROM python:3.10-slim

# OpenCV (pulled by ultralytics) needs these system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# HF runs the container as a non-root user (uid 1000); keep caches writable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
