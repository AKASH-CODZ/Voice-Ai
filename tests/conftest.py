from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

MODELS = REPO_ROOT / "models"


@pytest.fixture(scope="session")
def sample_rate() -> int:
    return 16_000


@pytest.fixture(scope="session")
def speech_audio(sample_rate: int) -> np.ndarray:
    """Real speech, not a synthetic tone.

    This matters: a harmonic stack that *looks* like speech on a spectrogram
    scores ~0.003 on Silero, identical to what a broken wrapper returns. Testing
    the VAD against synthetic audio would have hidden the v5 context bug
    completely — the fixture and the bug produce the same number.

    macOS gets real speech from `say`. Elsewhere we skip rather than silently
    fall back to a fixture that cannot distinguish pass from fail.
    """
    wav = MODELS / "_test_speech.wav"
    if not wav.exists():
        if sys.platform != "darwin":
            pytest.skip("no `say` available to generate real speech (macOS only)")
        MODELS.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["say", "-o", str(wav), f"--data-format=LEI16@{sample_rate}",
             "Hello there. I want to practice for a backend engineering interview today."],
            check=True, capture_output=True,
        )

    with wave.open(str(wav)) as fh:
        pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


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
