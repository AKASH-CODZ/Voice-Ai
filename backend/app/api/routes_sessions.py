"""Session listing, transcript retrieval, export and deletion."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.api.deps import client_ip
from app.core import export
from app.core.database import db
from app.core.schemas import SessionSummary, TurnRecord

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    request: Request,
    mine: bool = Query(True, description="Restrict to sessions from this IP."),
    limit: int = Query(50, ge=1, le=200),
) -> list[SessionSummary]:
    user_id = None
    if mine:
        user_id = await db.upsert_user(client_ip(request))
    return await db.list_sessions(user_id=user_id, limit=limit)


@router.get("/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/turns", response_model=list[TurnRecord])
async def get_turns(session_id: str) -> list[TurnRecord]:
    if await db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await db.get_turns(session_id)


@router.get("/{session_id}/stats")
async def get_stats(session_id: str) -> dict:
    """Latency and flag aggregates. Deliberately NOT part of the markdown
    export — see docs/DECISIONS.md → D-05."""
    if await db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await db.session_stats(session_id)


@router.get("/{session_id}/export.md")
async def export_markdown(session_id: str) -> Response:
    """Download the transcript as `User:` / `AI:` markdown and nothing else."""
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    content = await export.render_markdown(db, session_id)
    filename = export.export_filename(session_id, session.mode.value, session.started_at)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{session_id}/export")
async def save_markdown(session_id: str) -> dict:
    """Write the transcript to EXPORT_DIR on the server."""
    try:
        path = await export.write_markdown(db, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found") from None
    return {"path": path}


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    """The manual wipe from the spec — memory persists until explicitly deleted."""
    if not await db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@router.delete("")
async def delete_my_data(request: Request) -> dict:
    """Erase every session and turn belonging to this IP."""
    user_id = await db.upsert_user(client_ip(request))
    removed = await db.delete_user_data(user_id)
    return {"deleted_user": user_id, "rows": removed}
