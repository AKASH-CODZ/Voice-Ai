from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

MODELS = REPO_ROOT / "models"
SPEECH_TEXT = (
    "Hello there. I want to practice for a backend engineering interview today."
)


@pytest.fixture(scope="session")
def sample_rate() -> int:
    return 16_000


def _read_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path)) as fh:
        pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def _write_wav_float32(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    from app.core.audio import float32_to_pcm16

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(float32_to_pcm16(audio))


def _generate_speech_wav(path: Path, sample_rate: int) -> None:
    """Synthesize real speech. A tone scores ~0.003 on Silero — the same
    number a broken v5 wrapper returns — so it cannot be the fallback.

    Prefer macOS `say` (fast, no model load). On Linux, where `say` does not
    exist, use Kokoro — the TTS the pipeline already ships — so the D-10
    tripwire actually runs on the deployment machine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    say = shutil.which("say")
    if say is not None:
        subprocess.run(
            [say, "-o", str(path), f"--data-format=LEI16@{sample_rate}", SPEECH_TEXT],
            check=True,
            capture_output=True,
        )
        return

    model = MODELS / "kokoro-v0_19.onnx"
    voices = MODELS / "voices.bin"
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        pytest.skip(
            "no speech fixture: install kokoro-onnx (requirements-local.txt) "
            "or run on macOS where `say` can synthesize the D-10 VAD tripwire"
        )
    if not model.exists() or not voices.exists():
        pytest.skip(
            "no speech fixture: run `make models` so Kokoro can synthesize "
            "the D-10 VAD tripwire (macOS `say` is not available here)"
        )

    from app.core.audio import resample
    from app.core.config import settings

    # CPU only: this fixture needs speech, not GPU TTS. Leaving ONNX_PROVIDER
    # alone would couple the D-10 tripwire to the D-14 CUDA preload.
    kokoro = Kokoro(str(model), str(voices))
    samples, rate = kokoro.create(
        SPEECH_TEXT, voice=settings.kokoro_voice, speed=1.0, lang="en-us"
    )
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if int(rate) != sample_rate:
        audio = resample(audio, int(rate), sample_rate)
    _write_wav_float32(path, audio, sample_rate)


@pytest.fixture(scope="session")
def speech_audio(sample_rate: int) -> np.ndarray:
    """Real speech, not a synthetic tone.

    This matters: a harmonic stack that *looks* like speech on a spectrogram
    scores ~0.003 on Silero, identical to what a broken wrapper returns. Testing
    the VAD against synthetic audio would have hidden the v5 context bug
    completely — the fixture and the bug produce the same number.

    Cached WAV first, then macOS `say`, then Kokoro. Skip only if none of
    those can produce real speech — never fall back to a tone.
    """
    wav = MODELS / "_test_speech.wav"
    if not wav.exists():
        _generate_speech_wav(wav, sample_rate)
    return _read_wav_float32(wav)


@pytest.fixture(scope="session")
def silence_audio(sample_rate: int) -> np.ndarray:
    """Comfortably longer than STALL_SILENCE_MS (4 s) so the stall watchdog has
    room to fire — at exactly 4 s the frame count lands one short."""
    rng = np.random.default_rng(1234)
    return (0.0005 * rng.standard_normal(sample_rate * 8)).astype(np.float32)


@pytest.fixture
def vad_available() -> None:
    if not (MODELS / "silero_vad.onnx").exists():
        pytest.skip("Silero VAD weights missing — run `python tools/download_models.py --vad-only`")


@pytest.fixture
def tmp_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    return path
