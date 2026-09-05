"""PCM conversion and resampling invariants."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio import (
    duration_ms, float32_to_pcm16, pcm16_to_float32, resample, rms_dbfs, to_wav_bytes,
)


def test_round_trip_preserves_signal():
    original = (np.sin(np.linspace(0, 40 * np.pi, 4096)) * 0.6).astype(np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert np.max(np.abs(original - restored)) < 1e-4


def test_clipping_does_not_wrap():
    """Without the clip, 1.02 wraps to a large NEGATIVE int16 and you hear a
    loud click instead of mild distortion."""
    loud = np.array([1.5, -1.5, 1.02, -1.02], dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(loud))

    # Sign must survive: a wrap turns +1.02 into a large negative value, which
    # is the audible click this clip exists to prevent.
    assert np.all(np.sign(restored) == np.sign(loud)), "sign flipped — the wrap bug is back"
    # Magnitude must saturate at full scale, not fold back toward zero.
    assert np.all(np.abs(restored) <= 1.0)
    assert np.all(np.abs(restored) > 0.99)


def test_bytes_length_matches_sample_count():
    audio = np.zeros(512, dtype=np.float32)
    assert len(float32_to_pcm16(audio)) == 512 * 2


@pytest.mark.parametrize("src,dst", [(16000, 24000), (24000, 16000), (48000, 16000)])
def test_resample_changes_length_proportionally(src: int, dst: int):
    audio = np.zeros(src, dtype=np.float32)
    out = resample(audio, src, dst)
    assert abs(len(out) - dst) <= 2
    assert out.dtype == np.float32


def test_resample_is_identity_at_same_rate():
    audio = np.linspace(-1, 1, 100, dtype=np.float32)
    assert resample(audio, 16000, 16000) is audio


def test_resample_preserves_a_tone():
    """A 440 Hz tone must still be 440 Hz after resampling — the check that
    catches an aliasing regression if soxr is ever dropped."""
    src, dst, freq = 48000, 16000, 440.0
    t = np.arange(src) / src
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)

    out = resample(tone, src, dst)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), 1 / dst)[np.argmax(spectrum)]
    assert abs(peak_hz - freq) < 5.0


def test_wav_header_is_well_formed():
    wav = to_wav_bytes(np.zeros(1600, dtype=np.float32), 16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) == 44 + 1600 * 2


def test_duration_and_loudness():
    assert duration_ms(np.zeros(16000, dtype=np.float32), 16000) == 1000.0
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) < -100
    assert -1 < rms_dbfs(np.ones(100, dtype=np.float32)) < 1
