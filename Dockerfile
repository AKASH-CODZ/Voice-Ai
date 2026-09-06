# Hugging Face Space entry point (sdk: docker). Keep in sync with
# docker/Dockerfile.cloud — this file exists because Spaces look for
# `Dockerfile` at the app root, not under docker/.
#
# Cloud / demo image — CPU only. Groq STT/LLM, Kokoro TTS on CPU.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r backend/requirements.txt \
    && pip install kokoro-onnx==0.4.9

COPY tools/download_models.py ./tools/download_models.py
RUN python tools/download_models.py

COPY backend ./backend
COPY tools ./tools
COPY app.py ./app.py

RUN useradd --create-home --uid 1000 echosync \
    && mkdir -p /app/data /app/models \
    && chown -R echosync:echosync /app
USER echosync

ENV PYTHONPATH=/app/backend \
    ENGINE_MODE=cloud \
    DATABASE_PATH=/app/data/echosync.db \
    EXPORT_DIR=/app/data/exports \
    SILERO_VAD_PATH=/app/models/silero_vad.onnx \
    KOKORO_MODEL_PATH=/app/models/kokoro-v0_19.onnx \
    KOKORO_VOICES_PATH=/app/models/voices.bin

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
