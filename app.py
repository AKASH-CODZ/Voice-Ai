#!/usr/bin/env python3
"""Hugging Face Space / cloud-demo entry.

The Docker Space uses `Dockerfile` (cloud profile) and does not import this
file. Running it directly is the zero-Docker fallback:

    GROQ_API_KEY=... python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ENGINE_MODE", "cloud")
os.environ.setdefault("CLOUD_TTS_PROVIDER", "kokoro-onnx")

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
