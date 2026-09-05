# EchoSync MCP Server

Exposes your EchoSync practice sessions to any MCP client (Claude Desktop,
Claude Code, Cursor…) as tools, so you can ask questions about your own
speaking practice without leaving the editor:

> "Pull up my last observation-mode session and tell me where I hesitated."

This is the piece the spec's *"the user can do the analysis themselves later,
elsewhere"* requirement points at. The transcript export stays clean and
analysis-free by design; this server is the "elsewhere".

It reads the same SQLite database the backend writes. Because that database
runs in WAL mode, the server can read while a live session is being recorded —
no locking, no copy.

## Install

Requires `mcp >= 2.0`.

```bash
pip install -e ./mcp
```

Note that `mcp` pulls `starlette 1.x`, which requires `fastapi >= 0.141` in the
backend — the pinned `backend/requirements.txt` already accounts for this. If
you would rather keep the two fully decoupled, give this server its own venv:
it talks to SQLite directly and imports nothing from the backend.

## Register with Claude Code

```bash
claude mcp add echosync -- python -m echosync_mcp.server
```

Or add to `~/.claude.json` / your client's MCP config:

```json
{
  "mcpServers": {
    "echosync": {
      "command": "python",
      "args": ["-m", "echosync_mcp.server"],
      "env": { "DATABASE_PATH": "/absolute/path/to/echosync-ai/data/echosync.db" }
    }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `list_sessions` | Recent sessions with mode, engine, turn count, duration |
| `get_transcript` | Full transcript, optionally as the clean `User:`/`AI:` markdown |
| `get_session_stats` | Latency percentiles, hesitation and correction counts |
| `search_transcripts` | Full-text search across every turn you have ever spoken |
| `hesitation_report` | Where you stalled, with the surrounding context |
| `compare_engines` | Local vs cloud latency, measured from your own sessions |
