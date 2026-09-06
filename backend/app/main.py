"""EchoSync AI — FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import routes_sessions, routes_system, ws_voice
from app.core import logging as log_config
from app.core.config import settings
from app.core.database import db
from app.core.ollama import ensure_ollama
from app.engines.router import router as engine_router

log = logging.getLogger("echosync")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log_config.configure()
    await db.connect()

    # Kick Ollama so the first /api/health can say "starting" rather than
    # "unreachable", and so handshake does not pay the full wait.
    asyncio.create_task(asyncio.to_thread(ensure_ollama, 0.0))

    report = engine_router.probe()
    gpu = report.primary_gpu
    log.info("EchoSync AI %s starting (%s)", __version__, settings.echosync_env)
    log.info("CPU: %s · %d cores · %.1f GB RAM",
             report.cpu_model, report.cpu_logical_cores, report.total_ram_gb)
    if gpu:
        log.info("GPU: %s · %.1f/%.1f GB VRAM free · %s°C",
                 gpu.name, gpu.free_vram_gb, gpu.total_vram_gb, gpu.temperature_c)
    elif report.apple_silicon:
        log.info("GPU: Apple Silicon (Metal) — no CUDA; Whisper/Kokoro on CPU")
    else:
        log.info("GPU: none detected")
    if report.ollama_model:
        log.info("LLM: %s (%s)", report.ollama_model, report.ollama_status)
    else:
        log.info("Ollama: %s", report.ollama_status)
    log.info("Engine: %s — %s", report.recommended_engine, report.reason)

    # Engines are loaded lazily on the first connection rather than here. Two
    # reasons: the health endpoint must answer instantly for the frontend badge,
    # and a cloud-only deployment should never pay to import torch.
    try:
        yield
    finally:
        await engine_router.shutdown()
        await db.close()
        log.info("EchoSync AI stopped")


app = FastAPI(
    title="EchoSync AI",
    version=__version__,
    description=(
        "Hybrid local/cloud real-time English voice agent. "
        "Three interaction modes, sub-second target latency, 100% background logging."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_system.router, prefix="/api")
app.include_router(routes_sessions.router, prefix="/api")
app.include_router(ws_voice.router)


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse({
        "name": "EchoSync AI",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
        "websocket": "/ws/voice",
    })


if __name__ == "__main__":
    import uvicorn

    log_config.configure()
    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.echosync_env == "development",
    )
