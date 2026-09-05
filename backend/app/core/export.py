"""Markdown transcript export.

The spec is unusually strict about this file, and deliberately so: *"just a .md
of user: , Ai: and nothing more! no analysis just show what is heard"*.

So this module writes exactly that. No scores, no hesitation markers, no
latency, no correction counts — even though all of it is sitting right there in
SQLite. Those columns stay in the database for whatever you want to build on
top later (see docs/DECISIONS.md → D-05); they never leak into the transcript.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.core.config import settings
from app.core.database import Database
from app.core.schemas import Role

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


async def render_markdown(db: Database, session_id: str) -> str:
    """Return the transcript as a markdown string, or raise KeyError."""
    session = await db.get_session(session_id)
    if session is None:
        raise KeyError(session_id)

    turns = await db.get_turns(session_id)
    date = session.started_at.split("T")[0]

    header = [
        "# Session Transcript - EchoSync AI",
        "",
        f"Date: {date} | Mode: {session.mode.value.capitalize()}",
    ]
    if session.topic:
        header.append(f"Topic: {session.topic}")
    header.append("")

    body: list[str] = []
    for turn in turns:
        speaker = "User" if turn.role is Role.USER else "AI"
        # Collapse newlines: a transcript line must stay one line so the
        # alternating User:/AI: shape survives any downstream parser.
        text = " ".join(turn.text.split())
        if text:
            body.append(f"{speaker}: {text}")

    if not body:
        body.append("_No speech was recorded in this session._")

    return "\n".join(header + body) + "\n"


async def write_markdown(db: Database, session_id: str) -> str:
    """Render and persist to ``EXPORT_DIR``. Returns the file path."""
    content = await render_markdown(db, session_id)
    session = await db.get_session(session_id)
    assert session is not None

    stamp = datetime.fromisoformat(session.started_at).strftime("%Y%m%d-%H%M%S")
    name = _UNSAFE_FILENAME.sub("-", f"echosync-{stamp}-{session.mode.value}-{session_id}")
    path = settings.export_dir / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def export_filename(session_id: str, mode: str, started_at: str) -> str:
    """Content-Disposition filename for a download response."""
    try:
        stamp = datetime.fromisoformat(started_at).strftime("%Y%m%d-%H%M%S")
    except ValueError:
        stamp = "session"
    return _UNSAFE_FILENAME.sub("-", f"echosync-{stamp}-{mode}-{session_id}") + ".md"
