"""PCM helpers shared by the transport, VAD, STT and TTS stages.

Everything on the wire is **PCM16LE mono**; everything inside the pipeline is
**float32 in [-1, 1]**. Converting in exactly one place keeps scaling bugs
(the classic 32767-vs-32768 off-by-one that clips loud speech) out of the
engine code.
"""

from __future__ import annotations

import io
import wave

import numpy as np

INT16_SCALE = 32768.0


def pcm16_to_float32(data: bytes | np.ndarray) -> np.ndarray:
    """Bytes or int16 array → float32 in [-1, 1]."""
    arr = np.frombuffer(data, dtype=np.int16) if isinstance(data, (bytes, bytearray, memoryview)) else data
    return (arr.astype(np.float32) / INT16_SCALE).copy()


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 in [-1, 1] → PCM16LE bytes, hard-clipped rather than wrapped.

    Without the clip, a sample at 1.02 wraps to a large negative int16 and you
    hear a loud click instead of mild distortion.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * (INT16_SCALE - 1)).astype("<i2").tobytes()


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample float32 mono. Uses `soxr` (VHQ) when available.

    soxr matters on the STT path: a naive linear interpolation aliases high
    frequencies into the band Whisper actually listens to, and word error rate
    goes up measurably. The linear fallback exists only so the package still
    imports on a machine without soxr.
    """
    if src_rate == dst_rate or audio.size == 0:
        return audio
    try:
        import soxr

        return soxr.resample(audio, src_rate, dst_rate, quality="VHQ").astype(np.float32)
    except ImportError:
        ratio = dst_rate / src_rate
        n_out = int(round(audio.shape[-1] * ratio))
        return np.interp(
            np.linspace(0.0, audio.shape[-1] - 1, n_out, dtype=np.float64),
            np.arange(audio.shape[-1], dtype=np.float64),
            audio.astype(np.float64),
        ).astype(np.float32)


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Wrap float32 audio in a WAV container — what the Groq STT API wants."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float32_to_pcm16(audio))
    return buf.getvalue()


def duration_ms(audio: np.ndarray, sample_rate: int) -> float:
    return 1000.0 * audio.shape[-1] / sample_rate


def rms_dbfs(audio: np.ndarray) -> float:
    """Loudness in dBFS — used for the frontend orb and to skip dead air."""
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))
