"""Read-only SQLite access for the MCP server.

Deliberately standalone: this module does NOT import the FastAPI app. The MCP
server should start in a client like Claude Desktop without pulling in
onnxruntime, numpy, groq and the rest of the backend's dependency tree — the
user might not even have the backend installed on the machine running the
client.

Every connection is opened read-only (`mode=ro`) so a buggy tool can never
corrupt a database that a live session is writing to.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "echosync.db"


def database_path() -> Path:
    return Path(os.environ.get("DATABASE_PATH", DEFAULT_DB)).expanduser().resolve()


def connect() -> sqlite3.Connection:
    path = database_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No EchoSync database at {path}. Run a session first, or set "
            f"DATABASE_PATH to point at your echosync.db."
        )
    # immutable=0 + mode=ro: we can read a database being actively written in
    # WAL mode, we just cannot write to it ourselves.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None
