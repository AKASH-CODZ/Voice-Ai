# EchoSync AI

A real-time English conversation agent that runs **entirely on your own GPU**,
and falls back to the cloud for anyone who doesn't have one — same interface,
same three modes, same logging, no setup.

Built for interview practice and spoken-English drilling. You talk, it talks
back, and it behaves differently depending on which mode you put it in.

```
┌─────────────────────────────────────────────────────────────┐
│  Next.js frontend — reactive orb, mode switcher, transcript  │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket · PCM16 in 16 kHz / out 24 kHz
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI orchestrator                                        │
│  Silero VAD → utterance segmenter → hesitation watchdog      │
│  sentence chunker → streamed TTS → background telemetry      │
│  SQLite (WAL) · IP-bound sessions · dynamic engine router    │
└───────────┬─────────────────────────────┬───────────────────┘
            │                             │
   [GPU, ≥6 GB VRAM free]        [no GPU / demo]
            ▼                             ▼
 ┌────────────────────────┐   ┌──────────────────────────┐
 │ Faster-Whisper (CUDA)  │   │ Groq whisper-large-v3    │
 │ Ollama Llama-3.2-3B    │   │ Groq llama-3.1-8b-instant│
 │ Kokoro-82M             │   │ Kokoro-82M (CPU)         │
 └────────────────────────┘   └──────────────────────────┘
```

## The three modes

The audio path, the engines and the state machine are identical across all
three. The only things that change are the system prompt and whether hesitation
is allowed to reach the model at all — which is exactly why adding a fourth mode
is a prompt file plus an enum member.

| Mode | What it does to you |
|---|---|
| **Casual** | Natural chat. Mistakes get repaired through cross-questions ("did you mean…?"), never called out. You just have a good conversation. |
| **Teaching** | Active tutor. Corrects gently, one fix at a time — and when you go quiet for 4 seconds it steps in with a specific hint instead of leaving you stranded. |
| **Observation** | Mock interview. Zero feedback. No praise, no correction, no hint when you stall. Everything is recorded silently for you to review after. |

Hesitation and correction flags are recorded in **every** mode. Observation mode
just never shows them to you — the UI hides them too, not only the agent.

## Quick start

```bash
cp .env.example .env
make setup        # venv + backend deps + frontend deps
make models       # Silero VAD (2 MB) + Kokoro TTS (330 MB)
make hw           # tells you whether this machine can run locally, and why not
```

Then either run locally:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
make backend      # :8000
make frontend     # :3000
```

…or set `GROQ_API_KEY` in `.env` and skip the model download entirely — the
router will send you to the cloud pipeline automatically.

### Docker

```bash
docker compose -f docker/docker-compose.yml --profile gpu   up --build   # local GPU
docker compose -f docker/docker-compose.yml --profile cloud up --build   # CPU / demo
```

## Hardware routing

At handshake the backend probes the machine and picks an engine, then tells you
in plain language why. The badge in the UI expands to show the actual
diagnostic — free VRAM, GPU temperature, whether Ollama answered.

| Condition | Route |
|---|---|
| CUDA GPU, ≥ 6 GB VRAM free, Ollama up | **local** |
| Apple Silicon + Ollama up | **local** (Metal for the LLM, CPU for STT/TTS) |
| GPU at or above the thermal limit | cloud — avoids throttled latency |
| < 6 GB VRAM free, or no GPU | cloud |
| Local engine fails to load | cloud, with the reason surfaced — the session never dies |

Override any of it with `ENGINE_MODE` or the badge's auto/local/cloud buttons.

## Measured latency

`make bench` runs five real turns and reports per-stage numbers **for your
machine**, then names the dominant stage and what to do about it.

On the development machine (Apple M4, **CPU-only** STT and TTS — not the target
hardware):

| Stage | Measured |
|---|---|
| STT (`base.en`) | 252 ms |
| LLM time-to-first-token | 164 ms |
| TTS (Kokoro) | 653 ms |
| **End to end** | **1180 ms** |

That is above the 800 ms target, and TTS on CPU is why. On a CUDA GPU the same
pipeline should land considerably lower — but that number is not measured here,
so it is not claimed here. Run `make bench` on your own hardware.

## What's actually hard about this

Most of the engineering is in places that don't show up in a feature list:

- **Silero v5 needs a 64-sample context prefix.** Its ONNX input is declared
  `[None, None]`, so a bare 512-sample window is accepted silently and returns
  ~0.003 for *everything* — including obvious speech. Nothing raises. The agent
  just never responds. See `docs/decisions.md` → D-10.
- **The mic is gated while the agent speaks.** Browser AEC removes most echo,
  but one leaked syllable transcribed as user speech starts a loop where the
  agent answers itself forever. Barge-in uses a much stricter threshold instead.
- **The first chunk of a reply is spoken before the sentence finishes.** Waiting
  for a terminator that's 200 characters away cost 2 seconds of dead air;
  force-flushing the opening clause at a comma cut mean latency by 35%.
- **Telemetry never touches the audio thread.** Logging is a non-blocking queue
  put; a slow SQLite commit can't add jitter to speech.
- **Pre-roll ring buffer.** By the time VAD is confident speech started, the
  first ~100 ms is gone — which is exactly where the plosive that distinguishes
  "pat" from "bat" lives.

## Transcript export

Strictly `User:` / `AI:` and nothing else, by design:

```markdown
# Session Transcript - EchoSync AI

Date: 2026-08-19 | Mode: Teaching
Topic: backend engineer interview

User: Hello there, I want to practice for a back-end engineering interview today.
AI: Great. Let's start with your background — what's your current role?
```

Hesitation flags, corrections and latency all stay in SQLite and never enter
this file. Query them separately via the API or the MCP server.

## MCP server

`mcp/` exposes your sessions to any MCP client, so you can ask questions about
your own practice without leaving your editor:

> "Pull up my last observation session and tell me where I hesitated."

```bash
claude mcp add echosync -- python -m echosync_mcp.server
```

Tools: `list_sessions`, `get_transcript`, `get_session_stats`,
`search_transcripts`, `hesitation_report`, `compare_engines`. All read-only.

## Layout

```
echosync-ai/
├── backend/app/
│   ├── core/        config, hardware probe, SQLite, schemas, audio, export
│   ├── engines/     base protocols, local (Whisper/Ollama/Kokoro), cloud (Groq), router
│   ├── pipeline/    VAD + segmenter, sentence chunker, state machine, orchestrator, telemetry
│   ├── api/         REST routes + the voice WebSocket
│   └── prompts/     one markdown file per mode
├── frontend/src/    Next.js app, audio worklet, orb, transcript
├── mcp/             MCP server (read-only, standalone)
├── tools/           hardware check, model downloader, latency bench
├── tests/           68 tests
└── docs/            overview · architecture · data-model · decisions · features/
```

## Tests

```bash
make test     # 68 tests
make lint     # ruff + tsc
```

The VAD suite tests against **real speech**, not synthetic tones — a synthetic
harmonic scores the same ~0.003 on Silero that a broken wrapper does, so testing
against one would hide the exact bug it's meant to catch.

## Requirements

- Python **3.11 or 3.12** (3.13+ has patchy wheel coverage for onnxruntime)
- Node 20+
- Optional: NVIDIA GPU with ≥ 6 GB free VRAM + Ollama, or a `GROQ_API_KEY`

---

**Status and open questions** live in [`docs/decisions.md`](docs/decisions.md).
Current work-in-progress state and next steps are in
[`../HANDOFF.md`](../HANDOFF.md); the historical build log is in
[`../logs/archive.md`](../logs/archive.md).
