"""EchoSync MCP server — query your own speaking practice from any MCP client.

Run over stdio:

    python -m echosync_mcp.server

Why this exists: the transcript the product exports is deliberately clean —
`User:` / `AI:` and nothing else, per the spec. All the analytical signal
(hesitation flags, correction flags, per-stage latency) stays in SQLite and is
never written into that file. This server is where that signal becomes useful,
so the artifact stays pure and the insight is still one question away.

Every tool is read-only; the database is opened with `mode=ro`.
"""

from __future__ import annotations

import statistics
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import Field

from echosync_mcp import db

mcp: MCPServer = MCPServer(
    name="echosync",
    version="0.1.0",
    instructions=(
        "Read-only access to EchoSync AI voice-practice sessions. Sessions have "
        "one of three modes: casual (natural chat), teaching (active correction), "
        "observation (silent mock interview). Turn-level flags record where the "
        "speaker hesitated and where the agent corrected them; per-turn latency "
        "is recorded for both the local-GPU and cloud engines."
    ),
)

Mode = Literal["casual", "teaching", "observation"]
Role = Literal["user", "assistant"]


# ── helpers ──────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank. With the handful of turns a practice session produces,
    # interpolation would imply precision that is not there.
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return round(ordered[index], 1)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key)]
    return round(statistics.fmean(values), 1) if values else None


# ── tools ────────────────────────────────────────────────────

@mcp.tool()
def list_sessions(
    limit: Annotated[int, Field(description="Maximum sessions to return.", ge=1, le=200)] = 20,
    mode: Annotated[Mode | None, Field(description="Filter to one interaction mode.")] = None,
) -> dict[str, Any]:
    """List recorded practice sessions, newest first.

    Returns mode, topic, engine used, turn count and timestamps.
    """
    sql = """
        SELECT s.session_id, s.mode, s.topic, s.engine, s.started_at, s.ended_at,
               (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turns
        FROM sessions s
    """
    params: tuple[Any, ...] = ()
    if mode:
        sql += " WHERE s.mode = ?"
        params = (mode,)
    sql += " ORDER BY s.started_at DESC LIMIT ?"
    params += (limit,)

    rows = db.query(sql, params)
    return {"count": len(rows), "sessions": rows}


@mcp.tool()
def get_transcript(
    session_id: Annotated[str, Field(description="Session id from list_sessions.")],
    include_metadata: Annotated[
        bool, Field(description="Also return per-turn hesitation flags and latency.")
    ] = False,
) -> dict[str, Any]:
    """Fetch one session's full transcript.

    By default returns the clean `User:` / `AI:` markdown exactly as the product
    exports it. Set include_metadata to also get the analytical columns that are
    deliberately kept out of that file.
    """
    session = db.query_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    if session is None:
        return {"error": f"No session {session_id!r}."}

    turns = db.query(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id ASC", (session_id,)
    )

    lines = [
        "# Session Transcript - EchoSync AI",
        "",
        f"Date: {session['started_at'].split('T')[0]} | Mode: {session['mode'].capitalize()}",
    ]
    if session["topic"]:
        lines.append(f"Topic: {session['topic']}")
    lines.append("")
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "AI"
        lines.append(f"{speaker}: {' '.join(turn['text'].split())}")

    result: dict[str, Any] = {"session": session, "markdown": "\n".join(lines)}
    if include_metadata:
        result["turns"] = turns
    return result


@mcp.tool()
def get_session_stats(
    session_id: Annotated[str, Field(description="Session id from list_sessions.")],
) -> dict[str, Any]:
    """Latency and behaviour statistics for one session.

    Includes p50/p95 end-to-end latency, per-stage breakdown, and counts of
    hesitations and corrections.
    """
    session = db.query_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    if session is None:
        return {"error": f"No session {session_id!r}."}

    turns = db.query("SELECT * FROM turns WHERE session_id = ?", (session_id,))
    e2e = [t["e2e_ms"] for t in turns if t["e2e_ms"]]

    def stage(key: str) -> dict[str, Any] | None:
        values = [t[key] for t in turns if t[key]]
        if not values:
            return None
        return {"mean_ms": round(statistics.fmean(values), 1), "p95_ms": _percentile(values, 95)}

    return {
        "session_id": session_id,
        "mode": session["mode"],
        "engine": session["engine"],
        "turns": len(turns),
        "user_turns": sum(1 for t in turns if t["role"] == "user"),
        "hesitations": sum(1 for t in turns if t["hesitation"]),
        "corrections": sum(1 for t in turns if t["correction_made"]),
        "latency": {
            "e2e_p50_ms": _percentile(e2e, 50),
            "e2e_p95_ms": _percentile(e2e, 95),
            "e2e_mean_ms": round(statistics.fmean(e2e), 1) if e2e else None,
            "stt": stage("stt_ms"),
            "llm": stage("llm_ms"),
            "tts": stage("tts_ms"),
        },
    }


@mcp.tool()
def search_transcripts(
    query: Annotated[str, Field(description="Substring to search for.")],
    role: Annotated[Role | None, Field(description="Restrict to one speaker.")] = None,
    limit: Annotated[int, Field(description="Maximum matches.", ge=1, le=200)] = 30,
) -> dict[str, Any]:
    """Search every turn ever spoken, across all sessions.

    Useful for finding when a topic last came up, or how a phrase was used.
    """
    sql = """
        SELECT t.turn_id, t.session_id, t.role, t.text, t.timestamp, s.mode
        FROM turns t JOIN sessions s ON s.session_id = t.session_id
        WHERE t.text LIKE ?
    """
    params: tuple[Any, ...] = (f"%{query}%",)
    if role:
        sql += " AND t.role = ?"
        params += (role,)
    sql += " ORDER BY t.timestamp DESC LIMIT ?"
    params += (limit,)

    rows = db.query(sql, params)
    return {"query": query, "matches": len(rows), "results": rows}


@mcp.tool()
def hesitation_report(
    session_id: Annotated[
        str | None, Field(description="Omit to report across all sessions.")
    ] = None,
    limit: Annotated[int, Field(description="Maximum entries.", ge=1, le=200)] = 25,
) -> dict[str, Any]:
    """Every turn where the speaker stalled, with surrounding context.

    This is the "what did I get stuck on" view. Each entry carries the two turns
    before and one after, because a stall is only interpretable next to the
    question that provoked it.
    """
    sql = """
        SELECT t.turn_id, t.session_id, t.text, t.timestamp, s.mode, s.topic
        FROM turns t JOIN sessions s ON s.session_id = t.session_id
        WHERE t.hesitation = 1
    """
    params: tuple[Any, ...] = ()
    if session_id:
        sql += " AND t.session_id = ?"
        params += (session_id,)
    sql += " ORDER BY t.timestamp DESC LIMIT ?"
    params += (limit,)

    flagged = db.query(sql, params)
    for row in flagged:
        row["context"] = db.query(
            """
            SELECT role, text FROM turns
            WHERE session_id = ? AND turn_id BETWEEN ? AND ?
            ORDER BY turn_id
            """,
            (row["session_id"], row["turn_id"] - 2, row["turn_id"] + 1),
        )

    return {"count": len(flagged), "hesitations": flagged}


@mcp.tool()
def compare_engines() -> dict[str, Any]:
    """Compare measured local-GPU vs cloud latency from your own sessions.

    Uses recorded turns rather than vendor claims.
    """
    # Do NOT filter on e2e_ms here. STT latency is recorded against the USER
    # turn while e2e/llm/tts land on the ASSISTANT turn, so requiring e2e_ms
    # silently drops every STT sample and reports stt_mean_ms as null.
    rows = db.query(
        """
        SELECT s.engine, t.e2e_ms, t.stt_ms, t.llm_ms, t.tts_ms
        FROM turns t JOIN sessions s ON s.session_id = t.session_id
        WHERE t.e2e_ms IS NOT NULL OR t.stt_ms IS NOT NULL
        """
    )
    if not rows:
        return {"note": "No latency samples recorded yet — run a session first."}

    summary: dict[str, Any] = {}
    for engine in sorted({r["engine"] for r in rows}):
        subset = [r for r in rows if r["engine"] == engine]
        e2e = [r["e2e_ms"] for r in subset if r["e2e_ms"]]
        summary[engine] = {
            "turns_sampled": len(subset),
            "reply_samples": len(e2e),
            "e2e_p50_ms": _percentile(e2e, 50),
            "e2e_p95_ms": _percentile(e2e, 95),
            "stt_mean_ms": _mean(subset, "stt_ms"),
            "llm_mean_ms": _mean(subset, "llm_ms"),
            "tts_mean_ms": _mean(subset, "tts_ms"),
        }
    return {"engines": summary}


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
