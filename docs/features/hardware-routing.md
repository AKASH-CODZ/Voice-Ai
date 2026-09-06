# Hardware routing

## Purpose
Pick local (GPU) vs cloud (Groq) engines per request, and tell the user in
plain language why — without ever crashing on a machine that lacks a GPU.

## Key paths
- `backend/app/core/hardware.py` — the probe: NVML VRAM/temp/utilization,
  CPU/RAM inventory, Apple Silicon detection, routing verdict.
- `backend/app/core/ollama.py` — start a loopback `ollama serve` if the
  daemon is down; pick the best instruct model that fits free VRAM/RAM
  after Whisper + Kokoro. Coder models are never selected (D-13, D-18).
- `backend/app/engines/router.py` — `acquire()` honours the verdict, waits
  for Ollama on handshake, and degrades to cloud if local load fails.
- `backend/app/core/config.py` — `resolved_whisper_device` /
  `resolved_whisper_compute_type`: degrade `cuda`/`int8_float16` to
  `cpu`/`int8` instead of throwing when the configured device isn't usable.
- `tools/check_hardware.py` — CLI entry point (`make hw`) that prints the
  same diagnostic the router uses, and will start Ollama if it can.

## Routing table
| Condition | Route |
|---|---|
| CUDA GPU, ≥3.5GB free VRAM, Ollama up, instruct model fits | local |
| Apple Silicon + Ollama up + instruct model | local (Metal for LLM, CPU for STT/TTS) |
| Ollama down but startable | local once it answers; badge says "starting" |
| GPU at/above thermal limit | cloud (avoids throttled latency) |
| <3.5GB free VRAM, or no GPU (and not Apple Silicon) | cloud |
| Ollama up but only coder models pulled | cloud, with the pull hint |
| Local engine fails to load | cloud, with the reason surfaced |

Override via `ENGINE_MODE` env var or the frontend badge's auto/local/cloud
buttons (the buttons work before a session starts).

## Model pick
LLM headroom is `free_vram - 0.8 GB` (Whisper ~0.5 + Kokoro ~0.3) on CUDA,
or `total_ram - 5 GB` on Apple Silicon (macOS "available" RAM is not
headroom — file cache and a loaded Ollama model make it look tiny). Preferred tag is
`llama3.2:3b-instruct-q4_K_M` when it fits; otherwise fall to 1.5B / 1B
instruct. An `OLLAMA_MODEL` pin that is a coder model is ignored.

## Two VRAM floors
- `MIN_FREE_VRAM_GB=3.5` — room to *load* the local stack on a new session.
- `DEGRADE_FREE_VRAM_GB=1.0` — mid-session OOM guard. After load, free VRAM
  is ~3 GB lower; using 3.5 here would bounce every healthy local session
  to Groq after the first turn.

## Invariants
- `decide_engine()` never raises — every failure path degrades to "cloud"
  with a reason string, never an unhandled exception mid-session.
- The Apple Silicon branch is additive, not a replacement for the NVML/CUDA
  path — see `../decisions.md` D-08 before touching either.
- `ollama serve` is only spawned when `OLLAMA_BASE_URL` is loopback. A
  compose service named `ollama` is never started from inside the backend.

## Known sharp edges
- **One machine.** CUDA and Apple Silicon are alternate probe results on
  *this* host. The author's RTX 5070 and M4 laptops were used only as
  separate test beds; they do not run as a pair (D-19).
- CUDA laptop (8 GB Blackwell, measured): NVML works, `MIN_FREE_VRAM_GB=3.5`,
  local pipeline **530 ms** E2E. See `../decisions.md` D-12, D-17.
- Torch must be `>=2.7.1` from the **cu128** index for Blackwell (sm_120).
  `2.6.0+cu126` installs and reports CUDA, then dies on the first op.
  See `../decisions.md` D-02.
- Auto-pull of a missing instruct model is *not* done — a 2 GB fetch on
  first paint would stall the badge. The reason string names the tag to pull.
