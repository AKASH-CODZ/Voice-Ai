"""VAD regression tests.

The headline test here (`test_detects_real_speech`) exists because the Silero
v5 wrapper shipped broken at first: without the 64-sample context prefix the
model returns ~0.003 for speech AND silence, raising nothing. The agent simply
never responds. These tests are the tripwire for that class of failure.

The speech fixture is real speech (macOS `say`, or Kokoro on Linux) — a
synthetic tone scores the same ~0.003 and would hide the bug. See conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from app.pipeline.vad import SILERO_CONTEXT, SileroVAD, UtteranceSegmenter

pytestmark = pytest.mark.usefixtures("vad_available")


def frames(audio: np.ndarray, size: int) -> list[np.ndarray]:
    return [audio[i:i + size] for i in range(0, len(audio) - size, size)]


def test_frame_size_matches_silero_window():
    """32 ms @ 16 kHz must be exactly 512 samples. Changing FRAME_MS without
    changing SAMPLE_RATE silently degrades detection quality."""
    assert settings.frame_samples == 512


def test_detects_real_speech(speech_audio):
    vad = SileroVAD()
    probs = np.array([vad(f) for f in frames(speech_audio, settings.frame_samples)])
    assert probs.mean() > 0.7, f"real speech scored {probs.mean():.3f} — is the context prefix wired up?"
    assert probs.max() > 0.9


def test_rejects_silence(silence_audio):
    vad = SileroVAD()
    probs = np.array([vad(f) for f in frames(silence_audio, settings.frame_samples)])
    assert probs.max() < 0.3, f"silence scored up to {probs.max():.3f}"


def test_speech_and_silence_are_separable(speech_audio, silence_audio):
    """The property that actually matters: the two distributions must not
    overlap. A wrapper bug collapses both to the same near-zero value, which
    each test above could still pass in isolation on a bad threshold."""
    vad_a, vad_b = SileroVAD(), SileroVAD()
    speech = np.array([vad_a(f) for f in frames(speech_audio, settings.frame_samples)])
    quiet = np.array([vad_b(f) for f in frames(silence_audio, settings.frame_samples)])
    assert speech.mean() - quiet.mean() > 0.5


def test_context_buffer_is_carried_and_reset():
    vad = SileroVAD()
    assert vad._context.shape == (1, SILERO_CONTEXT)
    vad(np.ones(settings.frame_samples, dtype=np.float32) * 0.1)
    assert np.any(vad._context != 0), "context must carry the previous window's tail"
    vad.reset()
    assert not np.any(vad._context), "reset must clear the context with the LSTM state"


def test_segmenter_emits_one_utterance(speech_audio, silence_audio):
    seg = UtteranceSegmenter(SileroVAD())
    n = settings.frame_samples
    stream = frames(silence_audio, n)[:5] + frames(speech_audio, n) + frames(silence_audio, n)[:40]

    utterances = [d.utterance for f in stream if (d := seg.push(f)).utterance is not None]
    assert len(utterances) == 1

    # Pre-roll and trailing silence are intentionally included, so the captured
    # audio is longer than the speech itself — never shorter.
    assert len(utterances[0]) > len(speech_audio) * 0.9


def test_stall_does_not_fire_during_speech(speech_audio):
    seg = UtteranceSegmenter(SileroVAD())
    seg.arm_stall_watchdog()
    stalls = [
        d.stalled_ms
        for f in frames(speech_audio, settings.frame_samples) * 2
        if (d := seg.push(f)).stalled_ms is not None
    ]
    assert stalls == [], "a talking user must never be flagged as stalled"


def test_stall_fires_once_on_silence(silence_audio):
    seg = UtteranceSegmenter(SileroVAD())
    seg.arm_stall_watchdog()
    stalls = [
        d.stalled_ms
        for f in frames(silence_audio, settings.frame_samples)
        if (d := seg.push(f)).stalled_ms is not None
    ]
    assert stalls == [settings.stall_silence_ms], "stall must fire exactly once per arming"


def test_stall_requires_arming(silence_audio):
    """Silence while the AI is still speaking is not hesitation."""
    seg = UtteranceSegmenter(SileroVAD())
    stalls = [
        d.stalled_ms
        for f in frames(silence_audio, settings.frame_samples)
        if (d := seg.push(f)).stalled_ms is not None
    ]
    assert stalls == []


def test_short_blip_is_discarded(silence_audio):
    """A cough clears the threshold for a frame or two; it must not cost a
    full STT round-trip."""
    seg = UtteranceSegmenter(SileroVAD())
    n = settings.frame_samples
    blip = (np.random.default_rng(7).standard_normal(n) * 0.5).astype(np.float32)
    stream = frames(silence_audio, n)[:3] + [blip, blip] + frames(silence_audio, n)[:40]
    utterances = [d.utterance for f in stream if (d := seg.push(f)).utterance is not None]
    assert utterances == []


def test_vad_is_fast_enough_for_realtime(speech_audio):
    import time

    vad = SileroVAD()
    chunk = frames(speech_audio, settings.frame_samples)
    start = time.perf_counter()
    for f in chunk:
        vad(f)
    per_frame_ms = (time.perf_counter() - start) / len(chunk) * 1000
    # Must be a small fraction of the 32 ms each frame represents, or VAD alone
    # eats the latency budget.
    assert per_frame_ms < settings.frame_ms * 0.15, f"{per_frame_ms:.2f} ms/frame is too slow"
