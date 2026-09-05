# Data model

Source of truth: `backend/app/core/database.py` (`SCHEMA` constant). Don't
paste the schema here again when it changes — update the pointer, not this
file, unless the shape below stops matching.

## Entities
- **users** — one row per client IP. `user_id` derived from IP; no password,
  no account creation flow (D-04). `display_name` is captured on first visit
  purely to label transcripts.
- **sessions** — one row per conversation. Belongs to a `user_id`, records
  `mode` (casual/teaching/observation), `topic`, `engine` (local/cloud) used,
  start/end timestamps.
- **turns** — one row per utterance, either role `user` or `assistant`.
  Carries `text`, `hesitation` / `correction_made` flags, and per-stage
  timings: `stt_ms`, `llm_ms`, `tts_ms`, `e2e_ms`.

## Relationships
```
users (1) ──< sessions (1) ──< turns
```
`ON DELETE CASCADE` both ways: deleting a user drops their sessions and
turns; deleting a session drops its turns.

## Gotchas
- **STT latency lands on the user turn; e2e/llm/tts land on the assistant
  turn.** A query filtering `WHERE e2e_ms IS NOT NULL` will silently drop
  every STT sample — this bit the MCP server's `compare_engines` tool once
  (`decisions.md` D-11's sibling bug, fixed).
- WAL mode means readers (transcript API, MCP server) never block on the
  telemetry writer — don't "fix" this by adding locking.
- The MCP server opens the DB `mode=ro`. If a new MCP tool needs to write
  anything, that's a design smell — write access belongs to the backend only.

## Storage locations
- `data/echosync.db` (+ `-wal` / `-shm` sidecars) — not source, don't hand-edit.
- `data/exports/` — generated `User:`/`AI:` markdown transcripts, no analysis
  fields (see `decisions.md` D-05).
