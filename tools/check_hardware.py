#!/usr/bin/env python3
"""Print the exact diagnostic the engine router uses.

Run this first on any new machine. It answers "will this thing run locally, and
if not, why not" without starting the server.

    python tools/check_hardware.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings  # noqa: E402
from app.core.hardware import probe  # noqa: E402


def _tick(ok: bool) -> str:
    return "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"


def main() -> int:
    report = probe()

    print("\n\033[1mEchoSync AI — hardware diagnostic\033[0m")
    print("─" * 62)

    print(f"  Platform      {report.platform}")
    print(f"  CPU           {report.cpu_model}")
    print(f"                {report.cpu_physical_cores} physical / "
          f"{report.cpu_logical_cores} logical cores")
    print(f"  RAM           {report.available_ram_gb:.1f} GB free "
          f"of {report.total_ram_gb:.1f} GB")

    print("\n\033[1m  GPU\033[0m")
    if report.gpus:
        for gpu in report.gpus:
            head = f"{gpu.free_vram_gb:.1f} GB free of {gpu.total_vram_gb:.1f} GB"
            print(f"    [{gpu.index}] {gpu.name}")
            print(f"         VRAM  {head}")
            if gpu.temperature_c is not None:
                print(f"         Temp  {gpu.temperature_c}°C "
                      f"(limit {settings.gpu_thermal_limit_c}°C)")
            if gpu.utilization_pct is not None:
                print(f"         Util  {gpu.utilization_pct}%")
            if gpu.driver_version:
                print(f"         Drv   {gpu.driver_version}")
    elif report.apple_silicon:
        print("    Apple Silicon — no CUDA, Ollama runs on Metal")
    else:
        print("    none detected")

    print("\n\033[1m  Preconditions\033[0m")
    print(f"    {_tick(report.cuda_available or report.apple_silicon)} GPU acceleration available")
    print(f"    {_tick(report.ollama_reachable)} Ollama reachable at {settings.ollama_base_url}")
    print(f"    {_tick(report.cloud_credentials)} GROQ_API_KEY set (cloud fallback)")

    vad = settings.silero_vad_path
    kokoro = settings.kokoro_model_path
    print(f"    {_tick(vad.exists())} Silero VAD weights ({vad.name})")
    print(f"    {_tick(kokoro.exists())} Kokoro weights ({kokoro.name})")

    print("\n\033[1m  Resolved engine config\033[0m")
    print(f"    Whisper   {settings.whisper_model} on "
          f"{settings.resolved_whisper_device} / {settings.resolved_whisper_compute_type}")
    print(f"    LLM       {settings.ollama_model}")
    print(f"    TTS       kokoro ({settings.kokoro_voice})")

    colour = "\033[32m" if report.recommended_engine == "local" else "\033[33m"
    print(f"\n\033[1m  Verdict\033[0m  {colour}{report.recommended_engine.upper()}\033[0m")
    print(f"    {report.reason}")

    missing = [p.name for p in (vad, kokoro) if not p.exists()]
    if missing:
        print(f"\n  \033[33m→ Missing weights: {', '.join(missing)}\033[0m")
        print("    Run: python tools/download_models.py")

    if not report.ollama_reachable and report.recommended_engine == "cloud":
        print("\n  \033[33m→ To run locally, start Ollama:\033[0m")
        print("    ollama serve")
        print(f"    ollama pull {settings.ollama_model}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
