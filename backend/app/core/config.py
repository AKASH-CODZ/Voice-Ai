"""Central configuration.

Every knob the pipeline exposes lives here so the latency-critical modules
never reach for ``os.environ`` at runtime. Values come from ``.env`` (see
``.env.example``) and are validated once at import time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Runtime ──────────────────────────────────────────────
    echosync_env: Literal["development", "production"] = "development"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ── Engine routing ───────────────────────────────────────
    engine_mode: Literal["auto", "local", "cloud"] = "auto"
    min_free_vram_gb: float = 6.0
    gpu_thermal_limit_c: int = 85

    # ── Local pipeline ───────────────────────────────────────
    whisper_model: str = "base.en"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b-instruct-q4_K_M"

    kokoro_model_path: Path = Path("./models/kokoro-v0_19.onnx")
    kokoro_voices_path: Path = Path("./models/voices.bin")
    kokoro_voice: str = "af_sky"
    kokoro_device: str = "cuda"

    silero_vad_path: Path = Path("./models/silero_vad.onnx")

    # ── Cloud pipeline ───────────────────────────────────────
    groq_api_key: str = ""
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "llama-3.1-8b-instant"
    cloud_tts_provider: Literal["edge", "kokoro-onnx"] = "kokoro-onnx"
    edge_tts_voice: str = "en-US-AriaNeural"

    # ── Audio tuning ─────────────────────────────────────────
    sample_rate: int = 16000
    frame_ms: int = 32
    end_of_utterance_silence_ms: int = 800
    stall_silence_ms: int = 4000
    preroll_ms: int = 320
    barge_in_enabled: bool = True

    # ── Storage ──────────────────────────────────────────────
    database_path: Path = Path("./data/echosync.db")
    export_dir: Path = Path("./data/exports")
    max_tracked_users: int = 5
    context_window_turns: int = 12

    # ── Derived ──────────────────────────────────────────────
    @property
    def resolved_whisper_device(self) -> str:
        """`WHISPER_DEVICE=cuda` on a machine with no CUDA is a crash, not a
        config error worth failing startup over — fall back to CPU."""
        if self.whisper_device == "cuda" and not _cuda_present():
            return "cpu"
        return self.whisper_device

    @property
    def resolved_whisper_compute_type(self) -> str:
        # float16 compute types are meaningless on CPU and CTranslate2 rejects
        # some of them outright.
        if self.resolved_whisper_device == "cpu" and "float16" in self.whisper_compute_type:
            return "int8"
        return self.whisper_compute_type

    @property
    def frame_samples(self) -> int:
        """Samples per VAD frame. 32 ms @ 16 kHz = 512, which is exactly
        what Silero v5 expects — do not change one without the other."""
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2  # int16 mono

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_cloud_credentials(self) -> bool:
        return bool(self.groq_api_key.strip())

    @field_validator("database_path", "export_dir", "kokoro_model_path",
                     "kokoro_voices_path", "silero_vad_path", mode="after")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    def ensure_dirs(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _cuda_present() -> bool:
    """Cheap one-shot CUDA check that does not import torch."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
