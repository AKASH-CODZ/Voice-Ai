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
    monkeypatch.setattr(settings, "min_free_vram_gb", 3.5)
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


def test_5070_with_36gb_free_routes_local():
    """Measured on the target laptop: 3.6–4.3 GB free, local budget ~3.3 GB.
    The old 6.0 GB floor routed this to cloud even though it fits."""
    gpu = GPUInfo(
        0, "NVIDIA GeForce RTX 5070 Laptop GPU", 8.0, 3.6, 4.4, 52, 3, "596.36",
    )
    engine, reason = decide_engine(make_report(gpu=gpu))
    assert engine == "local"
    assert "3.6" in reason


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


def test_ollama_starting_on_gpu_routes_local():
    engine, reason = decide_engine(make_report(ollama_status="starting"))
    assert engine == "local"
    assert "starting" in reason.lower()


def test_inventoried_coder_only_routes_cloud():
    report = make_report(
        ollama_status="up",
        ollama_models=["qwen2.5-coder:1.5b"],
        ollama_model=None,
        ollama_pick_reason=(
            "Ollama is up but no instruct model is pulled "
            "(found qwen2.5-coder:1.5b) — routing to Groq."
        ),
    )
    engine, reason = decide_engine(report)
    assert engine == "cloud"
    assert "instruct" in reason.lower() or "coder" in reason.lower()


def test_should_degrade_ignores_the_load_floor(monkeypatch):
    """2 GB free is below MIN_FREE_VRAM_GB=3.5 but above the 1 GB OOM guard.
    A live local session must not bounce to Groq just because weights loaded."""
    from app.core import hardware as hw
    from app.core.config import settings
    from app.engines.router import EngineRouter

    gpu = GPUInfo(
        0, "NVIDIA GeForce RTX 5070 Laptop GPU", 8.0, 2.0, 6.0, 52, 3, "596.36",
    )
    monkeypatch.setattr(hw, "probe", lambda: make_report(gpu=gpu))
    monkeypatch.setattr(settings, "degrade_free_vram_gb", 1.0)
    monkeypatch.setattr(settings, "min_free_vram_gb", 3.5)
    should, _ = EngineRouter().should_degrade()
    assert should is False


def test_should_degrade_fires_near_oom(monkeypatch):
    from app.core import hardware as hw
    from app.engines.router import EngineRouter

    gpu = GPUInfo(
        0, "NVIDIA GeForce RTX 5070 Laptop GPU", 8.0, 0.4, 7.6, 52, 3, "596.36",
    )
    monkeypatch.setattr(hw, "probe", lambda: make_report(gpu=gpu))
    should, reason = EngineRouter().should_degrade()
    assert should is True
    assert "0.4" in reason


def test_health_payload_includes_model_fields():
    report = make_report(
        ollama_status="up",
        ollama_model="llama3.2:3b-instruct-q4_K_M",
        ollama_models=["llama3.2:3b-instruct-q4_K_M"],
        whisper_device="cpu",
    )
    payload = report.to_dict()
    assert payload["ollama_model"] == "llama3.2:3b-instruct-q4_K_M"
    assert payload["ollama_models"] == ["llama3.2:3b-instruct-q4_K_M"]
    assert payload["ollama_status"] == "up"
    assert payload["whisper_device"] == "cpu"
