"""Ollama auto-start and instruct-model pick (D-18)."""

from __future__ import annotations

import httpx
import pytest

from app.core.ollama import (
    ensure_ollama,
    is_instruct_candidate,
    llm_budget_gb,
    pick_from_models,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_ollama_state():
    reset_for_tests()
    yield
    reset_for_tests()


def test_coder_models_are_never_candidates():
    assert not is_instruct_candidate("qwen2.5-coder:1.5b")
    assert not is_instruct_candidate("qwen2.5-coder:7b")
    assert not is_instruct_candidate("codellama:7b")
    assert is_instruct_candidate("llama3.2:3b-instruct-q4_K_M")
    assert is_instruct_candidate("qwen2.5:1.5b-instruct")


def test_pick_prefers_3b_when_it_fits():
    name, reason = pick_from_models(
        ["qwen2.5-coder:1.5b", "llama3.2:3b-instruct-q4_K_M"],
        budget_gb=4.0,
        preferred="llama3.2:3b-instruct-q4_K_M",
    )
    assert name == "llama3.2:3b-instruct-q4_K_M"
    assert reason.endswith(".")


def test_pick_falls_to_15b_when_3b_is_too_big():
    name, reason = pick_from_models(
        ["llama3.2:3b-instruct-q4_K_M", "qwen2.5:1.5b-instruct"],
        budget_gb=1.6,
        preferred="llama3.2:3b-instruct-q4_K_M",
    )
    assert name == "qwen2.5:1.5b-instruct"
    assert "1.5" in reason or "1.6" in reason


def test_pick_never_selects_coder_even_if_only_option():
    name, reason = pick_from_models(
        ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b"],
        budget_gb=8.0,
        preferred="qwen2.5-coder:1.5b",
    )
    assert name is None
    assert "coder" in reason.lower() or "instruct" in reason.lower()
    assert reason.endswith(".")


def test_pick_ignores_coder_pin_when_instruct_is_pulled():
    name, _ = pick_from_models(
        ["qwen2.5-coder:1.5b", "llama3.2:3b-instruct-q4_K_M"],
        budget_gb=4.0,
        preferred="qwen2.5-coder:1.5b",
    )
    assert name == "llama3.2:3b-instruct-q4_K_M"


def test_vram_budget_subtracts_whisper_and_kokoro():
    assert llm_budget_gb(6.1, 16.0, use_vram=True) == pytest.approx(5.3)


def test_apple_silicon_budget_uses_total_ram_not_available():
    """4.3 GB 'available' on a 16 GB Mac is file cache + a loaded model, not
    headroom. Using it would route a working M4 to Groq."""
    assert llm_budget_gb(None, 4.3, use_vram=False, total_ram_gb=16.0) == pytest.approx(11.0)


def test_ensure_starts_daemon_then_sees_tags(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "llama3.2:3b-instruct-q4_K_M"}]}

    def fake_get(url, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("down")
        return _Resp()

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("app.core.ollama.shutil.which", lambda _: "/usr/bin/ollama")
    monkeypatch.setattr(
        "app.core.ollama.subprocess.Popen", lambda *a, **k: _Proc(),
    )
    monkeypatch.setattr("app.core.ollama.time.sleep", lambda _s: None)

    snap = ensure_ollama(timeout_s=2)
    assert snap.status == "up"
    assert "llama3.2:3b-instruct-q4_K_M" in snap.models
    assert snap.started_by_us is True


def test_ensure_does_not_spawn_when_base_url_is_remote(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama:11434")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")))
    spawned = {"n": 0}
    monkeypatch.setattr(
        "app.core.ollama.subprocess.Popen",
        lambda *a, **k: spawned.__setitem__("n", spawned["n"] + 1),
    )
    monkeypatch.setattr("app.core.ollama.shutil.which", lambda _: "/usr/bin/ollama")

    snap = ensure_ollama(timeout_s=0)
    assert snap.status == "down"
    assert spawned["n"] == 0
    assert snap.error and "loopback" in snap.error


@pytest.mark.asyncio
async def test_ollama_load_raises_on_missing_model():
    from app.engines.local import OllamaLLM

    class _Resp:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("404 must not be swallowed")

    class _Client:
        async def post(self, *a, **k):  # noqa: ARG002
            return _Resp()

    llm = OllamaLLM(model="missing-tag")
    llm._client = _Client()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="not pulled"):
        await llm.load()


def test_sticky_failure_is_cuda_not_connect():
    from app.engines.router import _is_sticky_local_failure

    assert _is_sticky_local_failure(ImportError("no torch"))
    assert _is_sticky_local_failure(RuntimeError("no kernel image is available"))
    assert not _is_sticky_local_failure(ConnectionError("ollama down"))
    assert not _is_sticky_local_failure(RuntimeError("Ollama model x is not pulled"))
