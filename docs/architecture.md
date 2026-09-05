# Architecture

## Pieces and how they talk
- **Frontend** (`frontend/src/`, Next.js) captures mic audio via an audio
  worklet, opens a binary WebSocket to the backend, plays back streamed TTS,
  and renders the orb/transcript/engine badge.
- **Transport** (`backend/app/pipeline/` boundary + `api/ws_voice.py`) is a
  native WebSocket carrying PCM16LE (16 kHz in, 24 kHz out) on the binary
  channel and a discriminated JSON envelope on the text channel — no base64.
  Everything downstream is behind a `Transport`-shaped interface so swapping
  transports (e.g. to LiveKit) touches one adapter, not the pipeline. See
  `decisions.md` D-01.
- **VAD + segmentation** (`pipeline/vad.py`) — Silero VAD decides
  speech/silence per frame; `UtteranceSegmenter` turns that into utterance
  boundaries, with a 320ms pre-roll ring buffer so onsets aren't clipped and
  a hesitation watchdog for Teaching mode's 4-second hint trigger.
- **Engine router** (`engines/router.py` + `core/hardware.py`) probes the
  machine (NVML VRAM/temp, Apple Silicon + Ollama liveness) and picks
  **local** (Faster-Whisper + Ollama + Kokoro) or **cloud** (Groq + Kokoro-onnx
  CPU) per request. `decide_engine()` is a pure function, unit-testable
  without a GPU. See `features/hardware-routing.md`.
- **Orchestrator** (`pipeline/orchestrator.py`) wires VAD → STT → LLM →
  sentence chunker → streamed TTS, force-flushing the first clause at a
  comma rather than waiting for a sentence terminator (cut mean latency 35%).
- **Telemetry** (`pipeline/telemetry.py`) is a non-blocking queue; nothing on
  the audio hot path ever awaits a DB write directly.
- **Persistence** (`core/database.py`) — SQLite in WAL mode, one connection.
  Readers (transcript API, MCP server) don't contend with the telemetry
  writer. See `data-model.md`.
- **MCP server** (`mcp/`) — separate process, opens the same SQLite file
  `mode=ro`. Read-only by construction, not just convention.

## Request/data flow (one turn)
1. Frontend streams mic PCM16 over the WebSocket binary channel.
2. VAD + segmenter decide an utterance ended → STT (local or cloud engine,
   per router) → text.
3. Text goes to the LLM with the mode's system prompt; response streams
   token-by-token into the sentence chunker.
4. Each completed chunk goes to TTS; audio streams back over the binary
   channel as it's generated — the reply is heard before the LLM has
   finished producing it.
5. Telemetry (STT/LLM/TTS/e2e timings, hesitation/correction flags) is
   queued to SQLite off the audio path, keyed to the turn.
6. On session end, `core/export.py` can produce the plain `User:`/`AI:`
   markdown transcript (no analysis fields — D-05).

## Where new code belongs
- New STT/LLM/TTS backend → `engines/`, implement `engines/base.py`'s
  protocol, register in `engines/router.py`.
- New mode → a prompt file in `prompts/` + an enum member in
  `pipeline/state.py`; the pipeline itself does not change.
- New REST endpoint → `api/routes_sessions.py` or `routes_system.py`
  depending on whether it's session data or system/health info.
- New MCP tool → `mcp/echosync_mcp/server.py`, read-only against the same DB.
