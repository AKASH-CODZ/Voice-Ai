"""SQLite persistence with WAL, plus the IP-bound session cache.

Design notes
------------
* **WAL mode** lets readers (the transcript API, the MCP server) run
  concurrently with the writer (the telemetry task) without blocking. At the
  1–5 user scale in the spec this is all the concurrency control we need.
* **Every write is awaited off the audio path.** Nothing in this module is
  called from the VAD/STT/TTS hot loop directly — the orchestrator hands
  records to ``TelemetryWriter`` (see ``pipeline/telemetry.py``), which drains
  a queue on its own task. Blocking here can never stall audio.
* **Users are identified by client IP**, per the spec. No auth, no passwords.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from app.core.config import settings
from app.core.schemas import Mode, Role, SessionSummary, TurnRecord

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,          -- derived from client IP
    ip_address   TEXT NOT NULL,
    display_name TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    mode        TEXT NOT NULL,
    topic       TEXT,
    engine      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    text             TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    hesitation       INTEGER NOT NULL DEFAULT 0,
    correction_made  INTEGER NOT NULL DEFAULT 0,
    stt_ms           REAL,
    llm_ms           REAL,
    tts_ms           REAL,
    e2e_ms           REAL
);

CREATE INDEX IF NOT EXISTS idx_turns_session   ON turns(session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user   ON sessions(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_ip        ON users(ip_address);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    """Thin async wrapper around one aiosqlite connection.

    One connection is correct here: SQLite serialises writes anyway, and WAL
    means our readers do not contend with it. A pool would add complexity and
    buy nothing at this scale.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = str(path or settings.database_path)
        self._db: aiosqlite.Connection | None = None

    # ── lifecycle ────────────────────────────────────────────

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        # WAL + NORMAL sync is the right trade for a local-first app: we keep
        # crash safety for committed transactions but skip an fsync per write.
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._db.execute("PRAGMA busy_timeout=5000;")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        log.info("SQLite ready at %s (WAL)", self._path)

    async def close(self) -> None:
        if self._db is not None:
            with contextlib.suppress(Exception):
                await self._db.commit()
            await self._db.close()
            self._db = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database.connect() was never awaited")
        return self._db

    # ── users ────────────────────────────────────────────────

    async def upsert_user(self, ip_address: str, display_name: str | None = None) -> str:
        """Identify a user by IP, creating the row on first contact.

        Returns the stable ``user_id``. Also enforces ``MAX_TRACKED_USERS`` by
        evicting the least-recently-seen user once the cap is exceeded, which
        keeps this a genuinely small, self-cleaning local database.
        """
        user_id = f"u_{uuid.uuid5(uuid.NAMESPACE_DNS, ip_address).hex[:12]}"
        now = _now()
        await self.conn.execute(
            """
            INSERT INTO users (user_id, ip_address, display_name, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                display_name = COALESCE(excluded.display_name, users.display_name)
            """,
            (user_id, ip_address, display_name, now, now),
        )
        await self.conn.commit()
        await self._evict_stale_users()
        return user_id

    async def _evict_stale_users(self) -> None:
        cur = await self.conn.execute("SELECT COUNT(*) AS n FROM users")
        row = await cur.fetchone()
        if row is None or row["n"] <= settings.max_tracked_users:
            return
        overflow = row["n"] - settings.max_tracked_users
        await self.conn.execute(
            """
            DELETE FROM users WHERE user_id IN (
                SELECT user_id FROM users ORDER BY last_seen ASC LIMIT ?
            )
            """,
            (overflow,),
        )
        await self.conn.commit()
        log.info("Evicted %d least-recently-seen user(s) to honour MAX_TRACKED_USERS", overflow)

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        cur = await self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    # ── sessions ─────────────────────────────────────────────

    async def create_session(
        self, user_id: str, mode: Mode, engine: str, topic: str | None = None
    ) -> str:
        session_id = f"s_{uuid.uuid4().hex[:16]}"
        await self.conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, mode, topic, engine, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, str(mode), topic, engine, _now()),
        )
        await self.conn.commit()
        return session_id

    async def end_session(self, session_id: str) -> None:
        await self.conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
            (_now(), session_id),
        )
        await self.conn.commit()

    async def update_session_mode(self, session_id: str, mode: Mode) -> None:
        await self.conn.execute(
            "UPDATE sessions SET mode = ? WHERE session_id = ?", (str(mode), session_id)
        )
        await self.conn.commit()

    async def update_session_engine(self, session_id: str, engine: str) -> None:
        await self.conn.execute(
            "UPDATE sessions SET engine = ? WHERE session_id = ?", (engine, session_id)
        )
        await self.conn.commit()

    async def get_session(self, session_id: str) -> SessionSummary | None:
        cur = await self.conn.execute(
            """
            SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turn_count
            FROM sessions s WHERE s.session_id = ?
            """,
            (session_id,),
        )
        row = await cur.fetchone()
        return _to_summary(row) if row else None

    async def list_sessions(self, user_id: str | None = None, limit: int = 50) -> list[SessionSummary]:
        sql = """
            SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turn_count
            FROM sessions s
        """
        params: tuple[Any, ...] = ()
        if user_id:
            sql += " WHERE s.user_id = ?"
            params = (user_id,)
        sql += " ORDER BY s.started_at DESC LIMIT ?"
        params += (limit,)

        cur = await self.conn.execute(sql, params)
        return [_to_summary(r) for r in await cur.fetchall()]

    async def latest_active_session(self, user_id: str) -> SessionSummary | None:
        """The IP-cache lookup: an unfinished session this user can resume."""
        cur = await self.conn.execute(
            """
            SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turn_count
            FROM sessions s
            WHERE s.user_id = ? AND s.ended_at IS NULL
            ORDER BY s.started_at DESC LIMIT 1
            """,
            (user_id,),
        )
        row = await cur.fetchone()
        return _to_summary(row) if row else None

    async def delete_session(self, session_id: str) -> bool:
        cur = await self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def delete_user_data(self, user_id: str) -> int:
        """The 'until delete is implemented' escape hatch from the spec."""
        cur = await self.conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        await self.conn.commit()
        return cur.rowcount

    # ── turns ────────────────────────────────────────────────

    async def add_turn(
        self,
        session_id: str,
        role: Role,
        text: str,
        *,
        hesitation: bool = False,
        correction_made: bool = False,
        stt_ms: float | None = None,
        llm_ms: float | None = None,
        tts_ms: float | None = None,
        e2e_ms: float | None = None,
    ) -> int:
        cur = await self.conn.execute(
            """
            INSERT INTO turns (session_id, role, text, timestamp, hesitation,
                               correction_made, stt_ms, llm_ms, tts_ms, e2e_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, str(role), text, _now(),
                int(hesitation), int(correction_made),
                stt_ms, llm_ms, tts_ms, e2e_ms,
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid or 0)

    async def get_turns(self, session_id: str, limit: int | None = None) -> list[TurnRecord]:
        sql = "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id ASC"
        params: tuple[Any, ...] = (session_id,)
        if limit:
            sql += " LIMIT ?"
            params += (limit,)
        cur = await self.conn.execute(sql, params)
        return [_to_turn(r) for r in await cur.fetchall()]

    async def recent_turns(self, session_id: str, n_turns: int) -> list[TurnRecord]:
        """Tail of the conversation, oldest-first — this is what gets replayed
        into the LLM context window each turn."""
        cur = await self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
            (session_id, n_turns),
        )
        rows = list(await cur.fetchall())
        return [_to_turn(r) for r in reversed(rows)]

    async def session_stats(self, session_id: str) -> dict[str, Any]:
        cur = await self.conn.execute(
            """
            SELECT COUNT(*)                                   AS turns,
                   SUM(hesitation)                            AS hesitations,
                   SUM(correction_made)                       AS corrections,
                   AVG(NULLIF(e2e_ms, 0))                     AS avg_e2e_ms,
                   AVG(NULLIF(stt_ms, 0))                     AS avg_stt_ms,
                   AVG(NULLIF(llm_ms, 0))                     AS avg_llm_ms,
                   AVG(NULLIF(tts_ms, 0))                     AS avg_tts_ms
            FROM turns WHERE session_id = ?
            """,
            (session_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}


def _to_summary(row: aiosqlite.Row) -> SessionSummary:
    return SessionSummary(
        session_id=row["session_id"],
        user_id=row["user_id"],
        mode=Mode(row["mode"]),
        topic=row["topic"],
        engine=row["engine"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        turn_count=row["turn_count"] if "turn_count" in row.keys() else 0,
    )


def _to_turn(row: aiosqlite.Row) -> TurnRecord:
    return TurnRecord(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        role=Role(row["role"]),
        text=row["text"],
        timestamp=row["timestamp"],
        hesitation=bool(row["hesitation"]),
        correction_made=bool(row["correction_made"]),
        stt_ms=row["stt_ms"],
        llm_ms=row["llm_ms"],
        tts_ms=row["tts_ms"],
        e2e_ms=row["e2e_ms"],
    )


# Module-level singleton, wired up in the FastAPI lifespan.
db = Database()


@contextlib.asynccontextmanager
async def lifespan_db() -> AsyncIterator[Database]:
    await db.connect()
    try:
        yield db
    finally:
        await db.close()
