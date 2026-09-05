"""Health, hardware diagnostics and engine routing preview."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.api.deps import client_ip
from app.core.config import settings
from app.core.database import db
from app.core.schemas import HealthResponse
from app.engines.router import router as engine_router

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness plus the routing verdict.

    The frontend calls this before opening the socket so it can render the
    engine badge and hardware readout on first paint rather than after the
    handshake — the diagnostic is the first thing a reviewer looks at.
    """
    report = engine_router.probe()
    return HealthResponse(
        version=__version__,
        engine_mode=settings.engine_mode,
        recommended_engine=report.recommended_engine,
        reason=report.reason,
        hardware=report.to_dict(),
    )


@router.get("/hardware")
async def hardware_report() -> dict:
    return engine_router.probe().to_dict()


@router.get("/whoami")
async def whoami(request: Request) -> dict:
    """Identify the caller by IP and hand back any resumable session.

    This is the IP cache from the spec: the browser hits it on load, and if
    this machine has an unfinished session we return it so the UI can offer
    'resume' instead of starting cold.
    """
    ip = client_ip(request)
    user_id = await db.upsert_user(ip)
    active = await db.latest_active_session(user_id)
    recent = await db.list_sessions(user_id=user_id, limit=10)
    return {
        "user_id": user_id,
        "ip_address": ip,
        "resumable_session": active.model_dump() if active else None,
        "recent_sessions": [s.model_dump() for s in recent],
    }


@router.post("/whoami/name")
async def set_display_name(request: Request, payload: dict) -> dict:
    name = (payload.get("display_name") or "").strip()[:64] or None
    ip = client_ip(request)
    user_id = await db.upsert_user(ip, display_name=name)
    return {"user_id": user_id, "display_name": name}
