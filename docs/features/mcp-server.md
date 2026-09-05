# MCP server

## Purpose
Let any MCP client (e.g. Claude Desktop) query your own practice sessions
without leaving the editor — "pull up my last observation session and tell
me where I hesitated."

## Key paths
- `mcp/echosync_mcp/server.py` — tool definitions, built on `MCPServer` with
  `@mcp.tool()` (schemas inferred from type hints — do not hand-write JSON
  schemas here, that was tried and deleted).
- `mcp/echosync_mcp/db.py` — read-only DB access.
- `mcp/pyproject.toml` — separate dependency set from `backend/`; `mcp` is
  deliberately not in `backend/requirements.txt` (see `../decisions.md` D-11).

## Tools (all read-only)
`list_sessions`, `get_transcript`, `get_session_stats`, `search_transcripts`,
`hesitation_report`, `compare_engines`.

## Invariants
- Opens SQLite `mode=ro`, always. A buggy tool must never be able to write
  to a database a live voice session is concurrently writing to.
- WAL mode means this works fine even while a session is actively recording
  — don't add locking to "fix" concurrent access.

## Known sharp edges
- `compare_engines` filtering `WHERE e2e_ms IS NOT NULL` will silently drop
  every STT sample, because STT timing lands on the user turn while
  e2e/llm/tts land on the assistant turn. Already fixed once — if a new
  aggregate query does the same thing, it will have the same silent bug.
  See `../data-model.md`.
- Run standalone: `make mcp` from `echosync-ai/`, or
  `claude mcp add echosync -- python -m echosync_mcp.server`. Consider
  giving it its own venv if you ever run it on a different machine than the
  voice backend — it only needs SQLite, nothing else from `backend/`.
