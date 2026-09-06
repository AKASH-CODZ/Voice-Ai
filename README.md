---
title: EchoSync AI
emoji: 🎙️
colorFrom: indigo
colorTo: rose
sdk: docker
app_port: 8000
pinned: false
short_description: Real-time spoken-English practice agent (cloud profile)
---

# EchoSync AI

A real-time English conversation agent that runs **entirely on your own
machine** when that machine can host it, and falls back to Groq when it
can't — same interface, same three modes, same logging.

Built for interview practice and spoken-English drilling. You talk, it talks
back, and it behaves differently depending on which mode you put it in.

One clone, one computer. There is no pair of machines to wire together.

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
   [GPU / Apple Silicon + Ollama]         [no GPU / demo]
            ▼                             ▼
 ┌────────────────────────┐   ┌──────────────────────────┐
 │ Faster-Whisper (CUDA)  │   │ Groq whisper-large-v3    │
 │ Ollama Llama-3.2-3B    │   │ Groq llama-3.1-8b-instant│
 │ Kokoro-82M             │   │ Kokoro-82M (CPU)         │
 └────────────────────────┘   └──────────────────────────┘
```

## Two ways in

GitHub cannot host the WebSocket voice path. A visitor gets **two doors**:

| Door | What happens |
|---|---|
| **Try live (Groq)** | **Render** free web services (mic → Groq + Edge TTS). Sleeps after 15 min idle. Hugging Face free = static page only. |
| **Run local (Ollama)** | Clone → `make setup && make models`. The probe recommends a model **for that machine** and auto-starts Ollama if it is installed but stopped. |

The live demo will **not** download an Ollama installer onto your laptop.
That would be a desktop installer, not a Space.

### Try live (Groq)

**Hosted demo (free):** Render — the only one of Railway / Render / Fly.io
that still offers a real free **web service** (Railway is a $5 trial then
paid; Fly.io dropped the free tier for new accounts).

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AKASH-CODZ/Voice-Ai)

1. Click the button (or **New → Blueprint** and pick this repo).
2. Paste `GROQ_API_KEY` when prompted. Do not commit it.
3. Wait for `echosync-api` and `echosync-web`. First request after idle
   takes ~1 minute (free instances sleep after 15 minutes).
4. Open the **echosync-web** URL and allow the microphone.

`render.yaml` at the repo root: Docker API (`docker/Dockerfile.render`,
Edge TTS so it fits 512 MB RAM) + Node frontend. SQLite is ephemeral.

**Hugging Face (free):** project page only —  
https://huggingface.co/spaces/Akash-8/EchoSync-AI  
Docker Spaces need HF PRO.

### Run local (Ollama)

```bash
git clone https://github.com/AKASH-CODZ/Voice-Ai.git
cd Voice-Ai
cp .env.example .env          # optional: GROQ_API_KEY for cloud fallback
make setup && make models     # venv + deps + Silero (2 MB) + Kokoro (330 MB)
make hw                       # same diagnostic the router uses
make backend                  # :8000
make frontend                 # :3000
```

Use **Python 3.11 or 3.12** (`python3` on some systems is 3.13/3.14 and
will fail `onnxruntime` / `faster-whisper` wheels).

- **NVIDIA GPU or Apple Silicon + Ollama** → local (Whisper + Ollama + Kokoro).
  If Ollama is installed but stopped, the backend starts it on loopback.
- **No GPU / no Ollama** → cloud, if `GROQ_API_KEY` is set.
- **Neither** → the badge says what is missing; the session does not crash.

We do **not** silently `ollama pull` a ~2 GB tag on first paint — that
would stall the badge. The badge names the tag; pull it once:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```

`ENGINE_MODE=auto` is the default. Override with the env var or the
auto / local / cloud buttons on the badge.

### Docker

GPU-less clone of the cloud path:

```bash
make up-cloud    # docker compose --profile cloud up --build
```

CUDA host (optional):

```bash
make up-gpu      # docker compose --profile gpu up --build
```

Equivalent compose invocations:

```bash
docker compose -f docker/docker-compose.yml --profile gpu   up --build
docker compose -f docker/docker-compose.yml --profile cloud up --build
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

## Hardware routing

At handshake the backend probes the machine and picks an engine, then tells you
in plain language why. The badge in the UI expands to show the actual
diagnostic — free VRAM, GPU temperature, whether Ollama answered.

| Condition | Route |
|---|---|
| CUDA GPU, ≥ 3.5 GB VRAM free, Ollama up | **local** |
| Apple Silicon + Ollama up | **local** (Metal for the LLM, CPU for STT/TTS) |
| GPU at or above the thermal limit | cloud — avoids throttled latency |
| < 3.5 GB VRAM free, or no GPU | cloud |
| Local engine fails to load | cloud, with the reason surfaced — the session never dies |

Override any of it with `ENGINE_MODE` or the badge's auto/local/cloud buttons.

## Measured latency

`make bench` runs five real turns on **the machine you ran it on**. Two
laptops were used only as independent test hosts (they never run as a pair):

| Stage | RTX 5070 Laptop (CUDA) | Apple M4 (CPU STT/TTS, Metal LLM) |
|---|---|---|
| STT (`base.en`) | 75 ms | 252 ms |
| LLM time-to-first-token | 35 ms | 164 ms |
| TTS (Kokoro) | 263 ms | 653 ms |
| **End to end** | **530 ms** (443–596, all 5 under 800) | 1180 ms |

Quote **530 ms** for a CUDA laptop in this class. Your numbers will differ;
run `make bench` locally.

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
├── tests/           pytest (93 tests at last count)
└── docs/            overview · architecture · data-model · decisions · features/
```

## Tests

```bash
make test     # pytest
make lint     # ruff + tsc
```

The VAD suite tests against **real speech**, not synthetic tones — a synthetic
harmonic scores the same ~0.003 on Silero that a broken wrapper does, so testing
against one would hide the exact bug it's meant to catch.

## Requirements

- Python **3.11 or 3.12** (3.13+ has patchy wheel coverage for onnxruntime)
- Node 20+
- Optional: NVIDIA GPU with ≥ 3.5 GB free VRAM + Ollama, or a `GROQ_API_KEY`

## Publishing

Source: [github.com/AKASH-CODZ/Voice-Ai](https://github.com/AKASH-CODZ/Voice-Ai)
(this directory is the public tree — not a parent workspace of session notes).

The Hugging Face Space uses this README's YAML frontmatter and the root
`Dockerfile` (cloud profile). Deploy the Space **only after**
`GROQ_API_KEY` is a Space secret — never commit the key.

---

Design choices live in [`docs/decisions.md`](docs/decisions.md). You do not
need the author's session notes to run the app.
