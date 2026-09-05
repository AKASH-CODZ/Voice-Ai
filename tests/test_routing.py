"""Engine routing policy.

`decide_engine` is deliberately a pure function of a HardwareReport so the
whole decision table is testable on a machine with no GPU at all.
"""

from __future__ import annotations

import pytest

from app.core.hardware import GPUInfo, HardwareReport, decide_engine


def make_report(**kwargs) -> HardwareReport:
    defaults = dict(cuda_available=True, ollama_reachable=True, cloud_credentials=True)
    gpu = kwargs.pop("gpu", GPUInfo(
        index=0, name="NVIDIA GeForce RTX 5070 Laptop GPU",
        total_vram_gb=8.0, free_vram_gb=7.2, used_vram_gb=0.8,
        temperature_c=52, utilization_pct=3, driver_version="560.94",
    ))
    report = HardwareReport(**{**defaults, **kwargs})
    report.gpus = [gpu] if gpu else []
    return report


@pytest.fixture(autouse=True)
def _auto_mode(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "engine_mode", "auto")
    monkeypatch.setattr(settings, "min_free_vram_gb", 6.0)
    monkeypatch.setattr(settings, "gpu_thermal_limit_c", 85)


def test_healthy_rtx5070_routes_local():
    engine, reason = decide_engine(make_report())
    assert engine == "local"
    assert "5070" in reason


def test_no_gpu_routes_cloud():
    engine, reason = decide_engine(make_report(cuda_available=False, gpu=None))
    assert engine == "cloud"
    assert "CUDA" in reason


def test_insufficient_vram_routes_cloud():
    """The interviewer-on-a-weak-laptop path from the spec."""
    gpu = GPUInfo(0, "NVIDIA GeForce GTX 1650", 4.0, 3.1, 0.9, 45, 0, "550.0")
    engine, reason = decide_engine(make_report(gpu=gpu))
    assert engine == "cloud"
    assert "3.1" in reason


def test_thermal_throttle_routes_cloud():
    gpu = GPUInfo(0, "RTX 5070", 8.0, 7.0, 1.0, 88, 99, "560.94")
    engine, reason = decide_engine(make_report(gpu=gpu))
    assert engine == "cloud"
    assert "88" in reason


def test_gpu_ready_but_ollama_down_routes_cloud():
    engine, reason = decide_engine(make_report(ollama_reachable=False))
    assert engine == "cloud"
    assert "Ollama" in reason


def test_apple_silicon_with_ollama_routes_local():
    report = make_report(cuda_available=False, gpu=None, apple_silicon=True, ollama_reachable=True)
    engine, reason = decide_engine(report)
    assert engine == "local"
    assert "Apple Silicon" in reason


def test_apple_silicon_without_ollama_routes_cloud():
    report = make_report(cuda_available=False, gpu=None, apple_silicon=True, ollama_reachable=False)
    engine, reason = decide_engine(report)
    assert engine == "cloud"
    assert "Ollama" in reason


def test_explicit_mode_overrides_hardware(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "engine_mode", "cloud")
    assert decide_engine(make_report())[0] == "cloud"

    monkeypatch.setattr(settings, "engine_mode", "local")
    assert decide_engine(make_report(cuda_available=False, gpu=None))[0] == "local"


def test_reason_is_always_human_readable():
    """The frontend renders this string verbatim in the engine badge."""
    for report in (
        make_report(),
        make_report(cuda_available=False, gpu=None),
        make_report(ollama_reachable=False),
    ):
        _, reason = decide_engine(report)
        assert reason and reason[0].isupper() and reason.endswith(".")
