# Decisions

Append-only. Add new entries at the bottom, in this format:

```markdown
## YYYY-MM-DD — <title>
Status: accepted | proposed | superseded
Decision: …
Why: …
Rejected: …
```

Migrated from the original `docs/DECISIONS.md` (13 entries, same numbering
kept below as titles for continuity). Several are marked "open" — they were
built under an assumption, not resolved, and still need Akash's sign-off.

---

## 2026-08-19 — D-01: Transport is native WebSocket, not LiveKit + Pipecat
Status: accepted (confirmed 2026-09-06 — keep native WS; re-open only if a public Space shows packet loss)
Decision: Binary WebSocket (PCM16LE) behind a `Transport` interface
(`pipeline/transport.py`), not LiveKit/Pipecat.
Why: The plan specified LiveKit, but on localhost there's no packet loss or
NAT — WebRTC's advantages don't apply, and LiveKit adds a 4th container, a
JWT token service, and TURN/STUN config for zero measured benefit at 1–5
users on loopback.
Rejected: LiveKit + Pipecat. Reconsider only if demoing over the public
internet (real packet loss), not localhost/tunnel. Reversal cost is low —
one adapter class against `Transport`, nothing else in the pipeline changes.

## 2026-08-19 — D-02: Torch pinned to 2.6.0 / cu126 for the RTX 5070
Status: superseded 2026-09-06 — 2.6.0+cu126 has no sm_120 kernels either
Decision: `torch==2.6.0`, cu126 wheel index.
Why: The 5070 is Blackwell (sm_120); cu121 wheels ship no sm_120 kernels →
`no kernel image is available for execution on the device` on first CUDA op.
Rejected: Default cu121 wheels. Watch for: if `check_hardware.py` sees the
GPU but Faster-Whisper throws a kernel error, this is the cause — fix the
index URL, not the code.

## 2026-09-06 — D-02 (closed): Torch 2.7.1+ / cu128 for Blackwell sm_120
Status: accepted
Decision: `torch>=2.7.1` from the **cu128** index
(`https://download.pytorch.org/whl/cu128`). First verified target: 2.7.1+cu128.
Why: Measured on the RTX 5070 Laptop: `torch==2.6.0+cu126` installs, reports
`cuda True`, then dies on the first CUDA op with
`no kernel image is available for execution on the device`. Its build only
ships sm_50–sm_90. Blackwell (sm_120) needs CUDA 12.8 wheels; PyTorch added
those in 2.7.0. xray_ml on the same machine already runs 2.9.1+cu128 (Python
3.10, unusable here because of the 3.11 floor).
Rejected: staying on 2.6.0/cu126 (D-02 original — insufficient, not just
cu121). Reversal is one pip install.

## 2026-08-19 — D-03: Kokoro TTS runs on CPU by default, not GPU
Status: superseded by the 2026-09-05 GPU measurement below
Decision: `KOKORO_DEVICE=cpu` via `onnxruntime` (the plan's `.env.example`
still lists `cuda`).
Why: `onnxruntime-gpu` pulls its own CUDA/cuDNN alongside Torch's; on an 8GB
card that duplication is real. CPU costs ~40–80ms more per chunk but frees
VRAM for Whisper + the LLM.
Rejected: GPU Kokoro (`onnxruntime-gpu`) as the default — the flag exists,
flip it after benchmarking on real hardware.

## 2026-08-19 — D-04: Sessions are IP-bound + localStorage, not accounts
Status: accepted (confirmed 2026-09-06 — keep IP + localStorage; no users table)
Decision: No auth. `user_id` derived from client IP, display name captured
once client-side for labeling only.
Why: Plan section 4 specifies IP-based session caching with no auth, for a
1–5 user scope; a stray line elsewhere ("after login") doesn't override that.
Rejected: Real accounts (users/credentials table, password hashing, session
tokens) — a day of extra work and meaningfully more attack surface for a
5-user tool. Revisit only if this stops being a single/small-group tool.

## 2026-08-19 — D-05: Markdown export is strictly `User:`/`AI:`, no analysis
Status: accepted
Decision: Transcript export has a header line plus alternating speaker
turns only. Hesitation/correction flags and timings stay in SQLite.
Why: Spec requires a clean export, but Observation mode's design implies an
end-of-session diagnostic report — resolved by keeping those two concerns
separate: the data is captured either way, just not in the `.md` file.
Rejected: Baking the diagnostic report into the `.md` export. Flagging in
case the report should ship as a separate generated file later.

## 2026-08-19 — D-06: Publish target — GitHub as source of truth
Status: accepted (Space not deployed — needs a Groq key as a Space secret)
Decision: GitHub is source of truth. Hugging Face Space config is in-tree:
`echosync-ai/README.md` YAML frontmatter (`sdk: docker`),
`echosync-ai/Dockerfile` (cloud profile), `echosync-ai/app.py` shim.
Why: HF Spaces free tier is CPU-only, so the Space can only run the cloud
profile. Rate limits on the Groq key become the demo ceiling.
Rejected: Shipping no Space config. Reversal is deleting those three files.
Do not deploy the Space until `GROQ_API_KEY` is set as a Space secret.

## 2026-08-19 — D-07: Cloud TTS default is `kokoro-onnx`, not EdgeTTS
Status: accepted
Decision: `CLOUD_TTS_PROVIDER=kokoro-onnx` (same 82M model as local, CPU).
Why: EdgeTTS returns MP3, which needs decoding somewhere (browser = second
audio path/bugs; Python = hard ffmpeg dependency). Kokoro-onnx emits float32
PCM directly, no network hop, no ffmpeg, and gives the cloud demo the same
voice as the local build.
Rejected: EdgeTTS as default — it's fully implemented and selectable (streams
MP3 into ffmpeg, decodes from stdout) if voice quality matters more than the
simplicity above; the cloud Docker image already installs ffmpeg for it.

## 2026-08-19 — D-08: Added an Apple Silicon local routing path
Status: accepted
Decision: `hardware.py` detects Apple Silicon and routes to **local** when
Ollama is reachable, instead of declaring any non-NVIDIA machine GPU-less.
`config.py` degrades `cuda`/`int8_float16` settings to `cpu`/`int8` instead
of throwing at model load.
Why: The plan's routing was CUDA-or-cloud only; that made the dev Mac (which
runs Ollama on Metal fine) unable to test locally without a Groq key.
Rejected: Leaving Apple Silicon cloud-only. This is a third route, additive
— the RTX 5070/NVML path is untouched. Revert by deleting the
`_is_apple_silicon` branch in `decide_engine()` if strict NVIDIA-or-cloud
scope is wanted for portfolio clarity.

## 2026-08-19 — D-09: Python floor is 3.11
Status: accepted
Decision: Built and tested against Python 3.11.12; 3.11/3.12 required.
Why: Code uses `StrEnum`, `datetime.UTC`, PEP 604 unions (3.11+). Newer
(3.13/3.14) has patchy `onnxruntime`/`faster-whisper` wheel coverage — this
dev machine's default `python3` is 3.14 and fails several installs.
Rejected: Targeting whatever `python3` resolves to. Install 3.11/3.12
explicitly on any new machine, including the Windows laptop.

## 2026-08-19 — D-10: Silero v5 needs a 64-sample context prefix (RESOLVED)
Status: accepted
Decision: Prepend the last 64 samples of the previous window to the current
512-sample window (576 total), carried across calls like LSTM state.
Why: Silero v5's ONNX input is declared `[None, None]`, so a bare 512-sample
window is silently accepted and returns ~0.003 for everything, including
real speech — the agent just never responds, with no error anywhere. Fixed
and verified (0.929 on speech / 0.004 on silence).
Rejected: n/a — this is the fix, not a rejected alternative. If you swap in
`silero_vad_16k_op15.onnx` or a v4 build, re-run `tests/test_vad.py` — the
context requirement differs between versions and the failure is silent.

## 2026-08-19 — D-11: Backend upgraded to FastAPI 0.141.1 for the MCP package
Status: accepted
Decision: `fastapi 0.141.1` + `starlette 1.6.0` (was `fastapi 0.115.6`).
Why: Installing `mcp` into the same venv pulled `starlette 1.6.0`, which is
incompatible with the pinned FastAPI and broke import
(`Router.__init__() got an unexpected keyword argument 'on_startup'`).
Rejected: Pinning `mcp` down instead — not attempted; the upgrade path was
verified (all REST endpoints 200, WebSocket handshake works, full suite
passes). Only cosmetic change: `len(app.routes)` reports 8 not 18 (sub-router
mounting), routing itself unaffected. `mcp` deliberately lives in
`mcp/pyproject.toml`, not `backend/requirements.txt` — give it its own venv
if you want the two fully decoupled (it only reads SQLite).

## 2026-08-19 — D-12: Sub-second latency target — measured, not assumed
Status: accepted on the RTX 5070 (2026-09-06); Mac remains CPU-bound
Decision: Report only measured numbers per machine; do not claim the plan's
500–700ms target until it's actually measured on target hardware.
Why: On the dev Mac (CPU-only STT/TTS, M4), measured end-to-end is 1180ms —
TTS (Kokoro, CPU ONNX) is the dominant cost at 653ms, not STT (252ms) or LLM
TTFT (164ms). `make bench` names the dominant stage per-machine.
Rejected: Claiming ~400–550ms on the 5070 as fact — that's an expectation,
not a measurement. If TTS is still dominant on real hardware, D-03 (Kokoro on
GPU) is the lever; if STT dominates instead, drop to `tiny.en`.

2026-09-06 measurement on the RTX 5070 (automatic path, no hand-set
`ONNX_PROVIDER`/`LD_LIBRARY_PATH`, `torch==2.7.1+cu128`,
`ENGINE_MODE=local KOKORO_DEVICE=cuda make bench`):

| stage | 5070 mean | M4 CPU mean |
|---|---|---|
| STT | 75 ms | 252 ms |
| LLM TTFT | 35 ms | 164 ms |
| TTS | 263 ms | 653 ms |
| E2E | **530 ms** (443–596, all 5 under 800) | 1180 ms |

TTS is still the dominant stage, now 2.5× faster than the Mac. The earlier
203 ms TTS / 450 ms E2E figure was the same GPU with env vars set by hand
and a different cuDNN. 530 ms is the number to quote for the automatic path.

## 2026-08-19 — D-13: Dev-machine local LLM is `qwen2.5-coder:1.5b`, not the spec's model
Status: accepted (open: pull the real model before judging quality)
Decision: All conversational testing on this Mac used `qwen2.5-coder:1.5b`
(only model pulled here); `.env.example` still correctly defaults to
`llama3.2:3b-instruct-q4_K_M` per the plan.
Why: A coder model is a poor conversational partner — it drifts toward
technical explanation and replies are stiffer than a general instruct model.
Latency numbers are still meaningful; personality is not representative.
Rejected: Judging conversation quality from what's been tested so far. Run
`ollama pull llama3.2:3b-instruct-q4_K_M` before making that call.

## 2026-09-05 — D-03 (closed): Kokoro TTS runs on GPU
Status: accepted (open: automatic `ONNX_PROVIDER` path unverified)
Decision: `KOKORO_DEVICE=cuda` + `onnxruntime-gpu`. Measured on the RTX 5070:
TTS 901 → 203 ms (4.4×), E2E 1163 → 450 ms, all 5 bench turns under the
800 ms target.
Why: CPU TTS was the bottleneck on both machines (D-12). GPU costs ~2.2 GB of
`nvidia-*` wheels on disk and VRAM inside an 8 GB card, and is what takes
this project under its latency target.
Rejected: keeping the CPU default after the bench. Reversal is
`KOKORO_DEVICE=cpu` and dropping `onnxruntime-gpu`.
Caveat: the 450 ms figure was measured with `ONNX_PROVIDER` /
`LD_LIBRARY_PATH` set by hand. The code that replaces those (D-14) has not
been confirmed end-to-end — the host went offline mid-run. See `todo.md` §2.

## 2026-09-05 — D-14: Four silent failures between `KOKORO_DEVICE=cuda` and actual GPU TTS
Status: accepted (automatic path verified 2026-09-06)
Decision: `KokoroTTS.load()` now (1) honors `kokoro_device`, (2) sets
`ONNX_PROVIDER=CUDAExecutionProvider`, (3) preloads `libcublas.so.12`,
`libcublasLt.so.12`, and `libcudnn.so.9` from the pip `nvidia-*` wheels with
`RTLD_GLOBAL`, (4) reads back `session.get_providers()` and warns if CUDA was
asked for but CPU engaged. `requirements-local.txt` pins
`onnxruntime-gpu==1.20.2` (1.20.1 does not exist). The preload is those three
sonames, not a glob of `nvidia/*/lib`.
Why: four independent failures, none of which raised:
1. `kokoro_device` was dead config — declared, logged, never passed on. It
   defaults to `cuda`, so startup claimed GPU while running CPU.
2. The old pin `onnxruntime-gpu==1.20.1` does not exist (1.20.0 and 1.20.2
   do), so the documented install path could not succeed.
3. `kokoro-onnx` 0.4.9 gates GPU on `find_spec("onnxruntime-gpu")`. Module
   names cannot contain hyphens, so it is always `None`. `ONNX_PROVIDER` is
   the only supported way past it.
4. cuDNN 9 was not on the dynamic-linker path (pip puts it under
   `site-packages/nvidia/*/lib`), so the CUDA provider failed to load and
   onnxruntime fell back to CPU silently. `get_available_providers()` still
   listed CUDA; only `session.get_providers()` told the truth.
Measured cost of getting this wrong on the 5070: TTS 203 ms GPU vs 901 ms
CPU. Globbing every `lib*.so` also pulled `libcudnn_engines_precompiled`
(~700 MB), reckless inside WSL2's 7.6 GB default RAM cap.
Rejected: requiring every caller to set `LD_LIBRARY_PATH`; globbing the
nvidia lib tree.

2026-09-06 follow-up: the three-soname preload was not enough.
`libonnxruntime_providers_cuda.so` DT_NEEDs the rest of cuDNN 9
(`libcudnn_adv`, `libcudnn_ops`, `libcudnn_cnn`, `libcudnn_graph`,
`libcudnn_heuristic`, `libcudnn_ext`, `libcudnn_engines_runtime_compiled`,
`libcudnn_engines_tensor_ir`, and `libcudnn_engines_precompiled`). Missing any
one of them → silent CPU fallback. Preloading that full set (still no glob)
with `KOKORO_DEVICE=cuda` alone, no env vars: CUDA provider engaged,
TTS first-audio **193 ms**. RAM stayed at 6.6 GB available; the ~700 MB
precompiled engine was mmap'd, not an OOM.

## 2026-09-05 — D-15: Warm Whisper's CUDA path at startup
Status: accepted
Decision: `FasterWhisperSTT` gets a `_warm()` the same way Kokoro already
did — one dummy transcribe at load, not on the user's first utterance.
Why: First transcribe on the 5070 was 7165 ms vs ~100 ms steady state (71×);
the opening turn was 8.7 s end-to-end. Invisible on CPU, which is why it
survived Mac verification. After warm-up, measured on the 5070: turn-1 STT
7165 → 79 ms, turn-1 E2E 8666 → 1101 ms, STT mean 1513 → 84 ms.
Rejected: leaving CTranslate2's CUDA kernel selection on the first real turn.

## 2026-09-06 — D-16: VAD speech fixture falls back to Kokoro on Linux
Status: accepted
Decision: `tests/conftest.py` `speech_audio` generates real speech by cached
WAV, then macOS `say`, then Kokoro. Skip only if none of those can produce
real speech. Never synthesize a tone.
Why: On Linux `say` does not exist, so all five real-speech tests in
`test_vad.py` skipped — the D-10 silent-failure guard did not run on the
actual deployment platform. A harmonic stack scores ~0.003 on Silero, the
same number a broken v5 wrapper returns, so a tone would hide the bug.
Kokoro is already a project dependency and produces real speech: measured
0.886 mean / 1.000 max on Silero (thresholds 0.7 / 0.9); all 11 VAD tests
pass against that fixture.
Rejected: leaving the tests skipped on Linux; checking in a WAV (models/ is
gitignored); using a synthetic tone.

## 2026-09-06 — D-17: Local-routing VRAM floor is 3.5 GB, not 6.0
Status: accepted (open: freeing Windows-side VRAM is still the better fix)
Decision: `MIN_FREE_VRAM_GB=3.5` (config default and `.env.example`).
Why: Measured local budget on the 8 GB 5070 is ~3.3 GB (llama3.2:3b-q4 ~2.5 +
Whisper base.en int8 ~0.5 + Kokoro ~0.3). Observed free VRAM on that laptop
was 4.3 GB idle and 3.6 GB with Ollama recently used; 6.0 routed CLOUD every
time. 3.5 is the budget plus a small margin. 3.1 GB (GTX 1650 path) still
goes to cloud.
Rejected: leaving 6.0; 4.0 (still clouds at the 3.6 GB observation); 3.3 with
no margin. Reversal is one env var. Windows-side apps still hold VRAM that
WSL cannot enumerate — closing those remains the better long-term fix.

## 2026-09-06 — D-18: Auto-start Ollama and pick an instruct model by budget
Status: accepted
Decision: On process boot and handshake, if Ollama is down on a loopback
`OLLAMA_BASE_URL` and `ollama` is on PATH, spawn `ollama serve` and wait
for `/api/tags`. From the pulled tags, pick the best *instruct* model that
fits LLM headroom after Whisper + Kokoro (~0.8 GB VRAM, or total RAM
minus 5 GB on Apple Silicon — not `available`, which double-counts a
loaded model). Prefer `llama3.2:3b-instruct-q4_K_M`; fall to
1.5B / 1B rather than routing to cloud just because the 3B is tight.
Coder tags are never selected, including a stale `OLLAMA_MODEL` pin (D-13).
`OllamaLLM.load()` raises on a 404 so the router can fall back to cloud
before the user speaks. The engine badge shows engine, model, device, and
free VRAM/RAM on first paint. Mid-session degrade stays at 1 GB free
(`DEGRADE_FREE_VRAM_GB`), distinct from the 3.5 GB *load* floor.
Why: GPU-ready + Ollama stopped used to mean silent cloud. A coder pin in
`.env` used to override a pulled instruct model. Health only exposed
`ollama_reachable: bool`.
Rejected: auto-pulling a 2 GB tag on first paint (stalls the badge; the
reason names the tag instead); spawning `ollama serve` against a remote
compose URL; using the 3.5 GB load floor as the mid-session degrade
threshold (would bounce every healthy local session after weights load).

## 2026-09-06 — D-19: One clone, one machine; test laptops are not a cluster
Status: accepted
Decision: The public setup is `clone` → `make setup && make models` →
optional `GROQ_API_KEY` → `make backend` / `make frontend`. The hardware
probe (D-18) routes **that** computer to local or cloud. An Apple M4 and
an RTX 5070 were used as **independent test hosts** with different
hardware; they never share a session, a database, or a network role.
Why: Internal notes (SSH hostnames, "dev Mac vs target GPU") read as if
the product were a two-laptop system. A stranger cloning from GitHub
should not think they need both, or that the two must talk.
Rejected: documenting a split Mac-frontend / GPU-backend deployment as
the default. Reversal would be a real multi-host architecture, which this
repo does not implement.

## 2026-09-06 — D-01 / D-04 sign-off
Status: accepted
Decision: Keep native WebSocket (D-01). Keep IP + `localStorage`, no
accounts (D-04).
Why: Localhost and a single-machine clone do not need LiveKit. A public
Space can stay on the same Transport until packet loss is actually
observed; reversal is still one adapter. The tool is still 1–5 users on
one box — a users table is extra attack surface for no product gain.
Rejected: LiveKit before a Space exists; adding accounts before the
product is multi-tenant. Re-open D-01 only after a public demo shows
loss/NAT pain; re-open D-04 only if this stops being a small-group tool.

## 2026-09-07 — D-20: Render `fromService` host is not a browser origin
Status: accepted
Decision: `NEXT_PUBLIC_BACKEND_URL` must be the public URL
(`https://echosync-api.onrender.com`). `fromService.property: host` is
Render internal DNS (`echosync-api`); the browser cannot resolve it, so
the health badge shows "backend is down" while `/api/health` on the API
service is 200. The frontend origin helper expands a bare service name
to `<name>.onrender.com`. Do not restore `fromService` for this var.
Why: Blueprint env updates do not overwrite a dashboard value that was
already set from `fromService`. The live `echosync-web` bundle inlined
`"echosync-api"` and fetched `https://echosync-api/api/health`.
Rejected: relying on Next `/api` rewrites to the sibling Render service
(returned 500); telling public-demo visitors to `make backend`.

## 2026-09-07 — D-21: Groq LLM is `openai/gpt-oss-20b`
Status: accepted
Decision: Default `GROQ_LLM_MODEL` is `openai/gpt-oss-20b`. Groq shut down
`llama-3.1-8b-instant` for free/developer keys on 2026-08-16 (still listed
as Enterprise / Contact Sales). Live Render turns 404'd with
`model_not_found`. For gpt-oss, the cloud LLM sends `reasoning_effort=low`
and `include_reasoning=false` so the spoken path does not wait on or speak
the reasoning chain. `max_completion_tokens=220` replaces `max_tokens`.
Why: Groq's published replacement for the 8B Instant slot. Voice replies
must stay short; gpt-oss reasons by default.
Rejected: staying on `llama-3.1-8b-instant` (404 on the demo key);
`openai/gpt-oss-120b` as default (slower, more expensive, same 404-fix).

## 2026-09-07 — D-22: Prime the TTS AudioContext in the click, before await
Status: accepted
Decision: `AudioPlayback.prime()` constructs the 24 kHz output
`AudioContext` synchronously inside the Start-talking click, *before*
`getUserMedia`. `enqueue` also calls `resume()` if the context later
suspends.
Why: Chrome treats localhost as autoplay-exempt, so a context created
after `await getUserMedia()` still plays on the M4. The same code on
`https://echosync-web.onrender.com` leaves that second context
`suspended`: JSON transcripts arrive, PCM is scheduled into a silent
graph. Edge TTS ffmpeg now sets `-f mp3` on the stdin pipe (MP3 is not
probeable without it).
Rejected: a visible "tap to unmute" overlay as the primary fix.

## 2026-09-07 — D-23: Edge TTS 7.0.0 403s; pin 7.2.8
Status: accepted
Decision: Pin `edge-tts==7.2.8`. 7.0.0 handshakes `wss://speech.platform.bing.com`
with HTTP 403, so Render (`CLOUD_TTS_PROVIDER=edge`) sends no PCM while
JSON transcripts still stream. 7.2.8 produced MP3 locally (16 kB for a
short sentence). The frontend also speaks the assistant text via
`speechSynthesis` if a turn ends with zero PCM, so a Microsoft outage
is not silent.
Why: Live demo inaudible after Groq LLM was fixed. Local Groq key
confirmed gpt-oss-20b returns spoken replies; Groq Orpheus TTS requires
org terms acceptance, so it is not the Render default.
Rejected: Kokoro on Render free (512 MB, D-07/D-22); Groq Orpheus as
default without terms (org admin must accept at the Groq playground).
