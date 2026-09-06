"""Hardware diagnostics behind the local-vs-cloud routing decision.

The probe is deliberately cheap and fully non-fatal: on a machine with no
NVIDIA driver, no CUDA, or no ``pynvml`` at all, every call degrades to
"no usable GPU" rather than raising. That is what lets the exact same image
run on your laptop and on a reviewer's Hugging Face Space.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass, asdict, field
from typing import Any

import psutil

from app.core.config import settings
from app.core.ollama import (
    ensure_ollama,
    llm_budget_gb,
    pick_from_models,
    snapshot as ollama_snapshot,
)

log = logging.getLogger(__name__)

_BYTES_PER_GB = 1024**3


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    used_vram_gb: float
    temperature_c: int | None
    utilization_pct: int | None
    driver_version: str | None


@dataclass(slots=True)
class HardwareReport:
    """Everything the router and the frontend badge need, in one payload."""

    cuda_available: bool
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_model: str = ""
    cpu_physical_cores: int = 0
    cpu_logical_cores: int = 0
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    platform: str = ""
    apple_silicon: bool = False
    ollama_reachable: bool = False
    ollama_status: str = "down"  # up | starting | down | missing
    ollama_model: str | None = None
    ollama_models: list[str] = field(default_factory=list)
    ollama_pick_reason: str = ""
    whisper_device: str = ""
    cloud_credentials: bool = False

    # Verdict
    recommended_engine: str = "cloud"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [asdict(g) if not isinstance(g, dict) else g for g in self.gpus]
        return d

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None


def _probe_gpus() -> tuple[bool, list[GPUInfo]]:
    """Query NVML. Returns ``(cuda_available, gpus)``; never raises."""
    try:
        import pynvml  # provided by nvidia-ml-py
    except ImportError:
        log.debug("pynvml not installed — treating machine as GPU-less")
        return False, []

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 — driver absent is a normal case
        log.debug("NVML init failed (%s) — treating machine as GPU-less", exc)
        return False, []

    gpus: list[GPUInfo] = []
    try:
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
        except Exception:  # noqa: BLE001
            driver = None

        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:  # noqa: BLE001
                temp = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:  # noqa: BLE001
                util = None

            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_vram_gb=round(mem.total / _BYTES_PER_GB, 2),
                    free_vram_gb=round(mem.free / _BYTES_PER_GB, 2),
                    used_vram_gb=round(mem.used / _BYTES_PER_GB, 2),
                    temperature_c=temp,
                    utilization_pct=util,
                    driver_version=driver,
                )
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("NVML enumeration failed midway: %s", exc)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    return bool(gpus), gpus


def _is_apple_silicon() -> bool:
    """Apple Silicon has no CUDA but a very capable unified-memory GPU.

    Ollama drives it through Metal, and Whisper/Kokoro run acceptably on the
    performance cores. Without this check the router would tell an M-series Mac
    it has "no GPU" and demand a Groq key — which is plainly wrong on a machine
    that can serve the local pipeline fine.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _cpu_model() -> str:
    if platform.system() == "Windows":
        return platform.processor()
    if platform.system() == "Darwin":
        brand = shutil.which("sysctl")
        if brand:
            import subprocess

            try:
                out = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=1.0, check=False,
                )
                if out.returncode == 0:
                    return out.stdout.strip()
            except Exception:  # noqa: BLE001
                pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def probe(check_ollama: bool = True, ensure: bool = False) -> HardwareReport:
    """Run the full diagnostic and attach a routing verdict.

    ``ensure=True`` will spawn ``ollama serve`` if the daemon is down and
    this process is allowed to (loopback URL + binary on PATH). Health uses
    the cheap path; handshake and ``make hw`` pass ``ensure``.
    """
    cuda_available, gpus = _probe_gpus()
    vm = psutil.virtual_memory()

    if not check_ollama:
        snap_status, snap_models = "down", []
    elif ensure:
        snap = ensure_ollama()
        snap_status, snap_models = snap.status, snap.models
    else:
        snap = ollama_snapshot()
        snap_status, snap_models = snap.status, snap.models

    use_vram = bool(cuda_available and gpus)
    available_ram = round(vm.available / _BYTES_PER_GB, 2)
    total_ram = round(vm.total / _BYTES_PER_GB, 2)
    budget = llm_budget_gb(
        gpus[0].free_vram_gb if gpus else None,
        available_ram,
        use_vram=use_vram,
        total_ram_gb=total_ram,
    )
    picked, pick_reason = pick_from_models(
        snap_models, budget, preferred=settings.ollama_model,
    ) if snap_status == "up" else (None, "")

    report = HardwareReport(
        cuda_available=cuda_available,
        gpus=gpus,
        cpu_model=_cpu_model(),
        cpu_physical_cores=(physical := psutil.cpu_count(logical=False) or 0),
        cpu_logical_cores=(
            psutil.cpu_count(logical=True) or os.cpu_count() or physical or 0
        ),
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        platform=f"{platform.system()} {platform.release()}",
        apple_silicon=_is_apple_silicon(),
        ollama_reachable=snap_status in {"up", "starting"},
        ollama_status=snap_status,
        ollama_model=picked,
        ollama_models=list(snap_models),
        ollama_pick_reason=pick_reason,
        whisper_device=settings.resolved_whisper_device,
        cloud_credentials=settings.has_cloud_credentials,
    )

    engine, reason = decide_engine(report)
    report.recommended_engine = engine
    report.reason = reason
    return report


def decide_engine(report: HardwareReport) -> tuple[str, str]:
    """Pure routing policy — separated from the probe so it is unit-testable.

    Order matters: an explicit ``ENGINE_MODE`` override always wins, then we
    walk the local-engine preconditions and fall through to cloud with a
    human-readable reason the frontend can render verbatim.
    """
    if settings.engine_mode == "local":
        return "local", "ENGINE_MODE=local forces the local pipeline."
    if settings.engine_mode == "cloud":
        return "cloud", "ENGINE_MODE=cloud forces the Groq pipeline."

    gpu = report.primary_gpu

    if not report.cuda_available or gpu is None:
        # Apple Silicon: no CUDA, but Ollama runs on Metal and the CPU handles
        # Whisper + Kokoro. Treat it as local-capable rather than GPU-less.
        if report.apple_silicon:
            return _local_if_ollama(
                report,
                ready=(
                    "Apple Silicon with Ollama on Metal"
                    + (f" — {report.ollama_model}" if report.ollama_model else "")
                    + " (Whisper and Kokoro on CPU)."
                ),
                starting="Apple Silicon detected — starting the local Ollama model.",
            )
        return "cloud", "No CUDA device detected — routing to Groq."

    if gpu.free_vram_gb < settings.min_free_vram_gb:
        return "cloud", (
            f"{gpu.name} has only {gpu.free_vram_gb:.1f} GB VRAM free "
            f"(need {settings.min_free_vram_gb:.1f} GB) — routing to Groq."
        )

    if gpu.temperature_c is not None and gpu.temperature_c >= settings.gpu_thermal_limit_c:
        return "cloud", (
            f"{gpu.name} is at {gpu.temperature_c}°C, at or above the "
            f"{settings.gpu_thermal_limit_c}°C limit — routing to Groq to avoid throttling."
        )

    model_bit = f", {report.ollama_model}" if report.ollama_model else ", Ollama up"
    return _local_if_ollama(
        report,
        ready=f"{gpu.name} — {gpu.free_vram_gb:.1f} GB VRAM free{model_bit}.",
        starting=f"{gpu.name} is ready — starting the local Ollama model.",
    )


def _local_if_ollama(
    report: HardwareReport, *, ready: str, starting: str,
) -> tuple[str, str]:
    """Shared Ollama gate used by the CUDA and Apple Silicon branches."""
    if report.ollama_status == "starting":
        return "local", starting
    if report.ollama_status == "missing":
        return "cloud", (
            "Ollama is not installed — routing to Groq. "
            "Install Ollama to run locally."
        )
    if not report.ollama_reachable:
        return "cloud", (
            f"Ollama is unreachable at {settings.ollama_base_url} "
            "— routing to Groq. Start Ollama to use the local engine."
        )
    # Only refuse when we actually inventoried tags and none were usable.
    # Unit tests construct reports without a model list; those still route local.
    if report.ollama_models and not report.ollama_model:
        return "cloud", report.ollama_pick_reason or (
            "Ollama is up but no instruct model fits — routing to Groq."
        )
    return "local", ready
