# Audio pipeline

## Purpose
Turn a raw mic stream into utterance-level turns, and turn an LLM's streamed
text reply into audio that starts before the sentence finishes.

## Key paths
- `backend/app/core/audio.py` — the only place PCM16 ↔ float32 conversion
  happens; hard-clips before int16 cast, uses `soxr` VHQ resampling.
- `backend/app/pipeline/vad.py` — Silero VAD wrapper + `UtteranceSegmenter`.
- `backend/app/pipeline/chunker.py` — sentence chunker for streamed TTS;
  force-flushes the opening clause at a comma instead of waiting for a
  terminator.
- `backend/app/pipeline/state.py` — the per-session state machine (mode,
  hesitation watchdog, barge-in).
- `backend/app/pipeline/orchestrator.py` — wires all of the above together.

## Invariants
- A 512-sample Silero window alone is wrong — it silently returns ~0.003 for
  everything, speech included. The wrapper must prepend the last 64 samples
  of the previous window (576 total), carried like LSTM state. See
  `../decisions.md` D-10 before touching this file.
- Audio is hard-clipped before the int16 cast — an unclipped value >1.02
  wraps around to a loud click instead of clamping.
- The mic is gated while the agent speaks, specifically to avoid a
  feedback loop where one leaked syllable of TTS gets transcribed as user
  speech and the agent answers itself forever. Barge-in uses a stricter
  threshold than normal VAD for this reason — don't unify the two thresholds.
- A 320ms pre-roll ring buffer keeps the ~100ms of audio before VAD becomes
  confident speech started — that's where the plosive distinguishing e.g.
  "pat" from "bat" lives. Don't remove it to "simplify" the segmenter.

## Known sharp edges
- Testing must use **real speech samples**, not synthetic tones — a
  synthetic harmonic scores the same ~0.003 on Silero that a broken wrapper
  produces, so it would hide the exact bug it's meant to catch
  (`tests/test_vad.py`).
- Barge-in has been unit-tested but never exercised with a live microphone —
  see `HANDOFF.md` at the repo root.
- Per-turn timing (`stt_ms`/`llm_ms`/`tts_ms`/`e2e_ms`) attributes STT to the
  user turn and the rest to the assistant turn in SQLite — see
  `../data-model.md` before writing a query that assumes they're on the same
  row.
