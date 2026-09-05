# Hardware routing

## Purpose
Pick local (GPU) vs cloud (Groq) engines per request, and tell the user in
plain language why — without ever crashing on a machine that lacks a GPU.

## Key paths
- `backend/app/core/hardware.py` — the probe: NVML VRAM/temp/utilization,
  CPU/RAM inventory, Ollama liveness, Apple Silicon detection.
- `backend/app/engines/router.py` — `decide_engine()`, a pure function over
  the probe's output; unit-testable without a GPU present.
- `backend/app/core/config.py` — `resolved_whisper_device` /
  `resolved_whisper_compute_type`: degrade `cuda`/`int8_float16` to
  `cpu`/`int8` instead of throwing when the configured device isn't usable.
- `tools/check_hardware.py` — CLI entry point (`make hw`) that prints the
  same diagnostic the router uses.

## Routing table
| Condition | Route |
|---|---|
| CUDA GPU, ≥6GB free VRAM, Ollama up | local |
| Apple Silicon + Ollama up | local (Metal for LLM, CPU for STT/TTS) |
| GPU at/above thermal limit | cloud (avoids throttled latency) |
| <6GB free VRAM, or no GPU | cloud |
| Local engine fails to load | cloud, with the reason surfaced |

Override via `ENGINE_MODE` env var or the frontend badge's auto/local/cloud
buttons.

## Invariants
- `decide_engine()` never raises — every failure path degrades to "cloud"
  with a reason string, never an unhandled exception mid-session.
- The Apple Silicon branch is additive, not a replacement for the NVML/CUDA
  path — see `../decisions.md` D-08 before touching either.

## Known sharp edges
- Nothing has been run on the actual target GPU (RTX 5070) yet — only
  unit-tested against synthetic hardware reports. See `HANDOFF.md` at the
  repo root.
- Torch is pinned to `2.6.0`/cu126 specifically for Blackwell (sm_120)
  kernels — see `../decisions.md` D-02 before changing the torch version.
