"""Dynamic local-vs-cloud engine router.

Behaviour the spec asks for, in one place:

* At handshake, probe the GPU and pick an engine (``decide_engine``).
* Honour a manual override from the UI badge.
* If the local engine *fails to load* — bad driver, Ollama died, missing
  weights — do not kill the session. Fall back to cloud, tell the client why,
  and keep talking. A voice agent that returns a stack trace has failed at the
  only thing it does.
* Engines are cached and shared. Loading Whisper takes seconds and ~1 GB of
  VRAM; building a fresh one per WebSocket would exhaust an 8 GB card at the
  third connection.
"""

from __future__ import annotations

import asyncio
import logging

from app.core import hardware
from app.core.config import settings
from app.core.ollama import ensure_ollama
from app.core.schemas import EnginePreference
from app.engines.base import VoiceEngine
from app.engines.cloud import build_cloud_engine
from app.engines.local import build_local_engine

log = logging.getLogger(__name__)


class EngineRouter:
    """Owns the (at most two) engine instances for the whole process."""

    def __init__(self) -> None:
        self._local: VoiceEngine | None = None
        self._cloud: VoiceEngine | None = None
        self._lock = asyncio.Lock()
        self._local_failed_reason: str | None = None

    # ── probing ──────────────────────────────────────────────

    def probe(self, ensure: bool = False) -> hardware.HardwareReport:
        return hardware.probe(ensure=ensure)

    # ── acquisition ──────────────────────────────────────────

    async def acquire(
        self, preference: EnginePreference = EnginePreference.AUTO
    ) -> tuple[VoiceEngine, hardware.HardwareReport, str]:
        """Return ``(engine, report, reason)`` — never raises for routing
        reasons, only if *both* engines are unusable."""
        report = self.probe()

        if preference is EnginePreference.LOCAL:
            target, reason = "local", "Manual override: local engine requested."
        elif preference is EnginePreference.CLOUD:
            target, reason = "cloud", "Manual override: cloud engine requested."
        else:
            target, reason = report.recommended_engine, report.reason

        # CUDA/import/missing-weights failures are sticky: retrying a broken
        # stack on every connection just adds a multi-second stall. Ollama
        # being down at boot is not — it may have come up since (D-18).
        if (
            target == "local"
            and self._local_failed_reason
            and preference is not EnginePreference.LOCAL
        ):
            target = "cloud"
            reason = self._local_failed_reason

        if target == "local" and report.ollama_status in {"down", "starting"}:
            await asyncio.to_thread(ensure_ollama)
            report = self.probe()
            if preference is EnginePreference.AUTO:
                target, reason = report.recommended_engine, report.reason

        if target == "local":
            try:
                engine = await self._get_local(report)
                return engine, report, reason
            except Exception as exc:  # noqa: BLE001
                fail = (
                    f"Local engine failed to start ({type(exc).__name__}: {exc}) "
                    "— falling back to Groq."
                )
                log.error(fail, exc_info=True)
                if _is_sticky_local_failure(exc):
                    self._local_failed_reason = fail
                else:
                    self._local_failed_reason = None
                reason = fail

        engine = await self._get_cloud()
        return engine, report, reason

    async def _get_local(self, report: hardware.HardwareReport) -> VoiceEngine:
        async with self._lock:
            if self._local is None:
                gpu = report.primary_gpu
                engine = build_local_engine(
                    gpu.name if gpu else None,
                    ollama_model=report.ollama_model,
                )
                await engine.load()
                self._local = engine
            return self._local

    async def _get_cloud(self) -> VoiceEngine:
        async with self._lock:
            if self._cloud is None:
                engine = build_cloud_engine()
                await engine.load()
                self._cloud = engine
            return self._cloud

    # ── mid-session degradation ──────────────────────────────

    def should_degrade(self) -> tuple[bool, str]:
        """Check whether a live local session ought to move to cloud.

        Called between turns, never mid-utterance. Thermal throttling on a
        laptop 5070 is the realistic trigger: sustained inference pushes the
        GPU past its limit, clocks drop, and token generation quietly doubles
        in latency. Better to hand the LLM to Groq than to let the conversation
        turn sluggish.
        """
        if settings.engine_mode == "local":
            return False, ""

        report = hardware.probe()
        gpu = report.primary_gpu
        if gpu is None:
            return True, "GPU disappeared from NVML — switching to Groq."

        if gpu.temperature_c is not None and gpu.temperature_c >= settings.gpu_thermal_limit_c:
            return True, (
                f"GPU hit {gpu.temperature_c}°C (limit {settings.gpu_thermal_limit_c}°C) "
                "— switching the pipeline to Groq to avoid throttled latency."
            )

        # Mid-session floor is degrade_free_vram_gb (1.0), not the 3.5 GB
        # *load* floor: models are already resident, so free VRAM is ~3 GB
        # lower than at first route. Crossing 1 GB means we are one
        # allocation away from an OOM that would drop the call.
        if gpu.free_vram_gb < settings.degrade_free_vram_gb:
            return True, (
                f"Only {gpu.free_vram_gb:.1f} GB VRAM free — switching to Groq "
                "before we run out."
            )
        return False, ""


def _is_sticky_local_failure(exc: BaseException) -> bool:
    """True for failures that will not fix themselves without a restart."""
    if isinstance(exc, (ImportError, FileNotFoundError)):
        return True
    msg = str(exc).lower()
    return "cuda" in msg or "cudnn" in msg or "kernel image" in msg

    async def shutdown(self) -> None:
        for engine in (self._local, self._cloud):
            if engine is not None:
                await engine.unload()
        self._local = self._cloud = None


router = EngineRouter()
