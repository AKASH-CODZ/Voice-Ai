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
Status: accepted (open: confirm before a public-internet demo)
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
Status: accepted
Decision: `torch==2.6.0`, cu126 wheel index.
Why: The 5070 is Blackwell (sm_120); cu121 wheels ship no sm_120 kernels →
`no kernel image is available for execution on the device` on first CUDA op.
Rejected: Default cu121 wheels. Watch for: if `check_hardware.py` sees the
GPU but Faster-Whisper throws a kernel error, this is the cause — fix the
index URL, not the code.

## 2026-08-19 — D-03: Kokoro TTS runs on CPU by default, not GPU
Status: accepted (open: revisit after `make bench` on real GPU)
Decision: `KOKORO_DEVICE=cpu` via `onnxruntime` (the plan's `.env.example`
still lists `cuda`).
Why: `onnxruntime-gpu` pulls its own CUDA/cuDNN alongside Torch's; on an 8GB
card that duplication is real. CPU costs ~40–80ms more per chunk but frees
VRAM for Whisper + the LLM.
Rejected: GPU Kokoro (`onnxruntime-gpu`) as the default — the flag exists,
flip it after benchmarking on real hardware.

## 2026-08-19 — D-04: Sessions are IP-bound + localStorage, not accounts
Status: accepted
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
Status: proposed (open: Akash's call)
Decision: GitHub + Docker for the full local-GPU stack; a Hugging Face Space
running the cloud profile as a zero-setup live demo, if wanted.
Why: HF Spaces free tier is CPU-only, so it can only run the cloud profile
and needs a Groq key as a Space secret (rate limits become the demo ceiling).
Docker profiles already support this without a code change.
Rejected: Nothing yet — HF Space config (README frontmatter + `app.py` shim)
not built; ask before adding.

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
Status: proposed (open: re-measure once run on the RTX 5070)
Decision: Report only measured numbers per machine; do not claim the plan's
500–700ms target until it's actually measured on target hardware.
Why: On the dev Mac (CPU-only STT/TTS, M4), measured end-to-end is 1180ms —
TTS (Kokoro, CPU ONNX) is the dominant cost at 653ms, not STT (252ms) or LLM
TTFT (164ms). `make bench` names the dominant stage per-machine.
Rejected: Claiming ~400–550ms on the 5070 as fact — that's an expectation,
not a measurement. If TTS is still dominant on real hardware, D-03 (Kokoro on
GPU) is the lever; if STT dominates instead, drop to `tiny.en`.

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
