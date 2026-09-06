"""Ollama daemon lifecycle and instruct-model selection (D-18).

The router used to treat Ollama as a boolean: GET /api/tags in 0.75 s, else
cloud. That meant a GPU-ready machine with a stopped daemon (or only a coder
model pulled) silently left the local pipeline. This module:

* discovers an already-running daemon, or starts a prebuilt ``ollama serve``
  when the configured base URL is loopback and the binary is on PATH;
* lists pulled tags;
* picks the best *instruct* model that fits the free VRAM/RAM budget after
  Whisper + Kokoro. Coder models are never selected (D-13).

Auto-pull is intentionally not done here — a 2 GB fetch on first paint would
stall the badge. The pick reason tells the user which tag to pull.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence
from urllib.parse import urlparse

import httpx

from app.core.config import settings

log = logging.getLogger(__name__)

TAGS_TIMEOUT_S = 0.75
ENSURE_WAIT_S = 8.0
POLL_S = 0.25

# Measured on the 8 GB 5070: Whisper base.en ~0.5 GB + Kokoro ~0.3 GB.
WHISPER_KOKORO_VRAM_GB = 0.8
# Unified-memory reserve against *total* RAM (OS + app + Whisper/Kokoro).
# Do not subtract this from `available` — macOS file cache and a model that
# Ollama already loaded make "available" look like 4 GB on a working 16 GB
# Mac, which would bounce the local pipeline to Groq.
RAM_RESERVE_GB = 5.0

PREFERRED_PULL = "llama3.2:3b-instruct-q4_K_M"

_CODER_MARKERS = ("coder", "codellama", "starcoder", "deepseek-coder", "-code", "code-")
_EMBED_MARKERS = ("embed", "nomic-embed")


@dataclass(frozen=True)
class CatalogEntry:
    """One instruct option we are willing to run.

    ``tags`` are accepted Ollama names, preferred tag first. ``vram_gb`` is
    the estimated weight footprint (Q4), used against LLM headroom.
    Higher ``rank`` wins when several options fit.
    """

    tags: tuple[str, ...]
    vram_gb: float
    rank: int


# Best-that-fits, not biggest-available: 8B is omitted because we have not
# judged its conversational quality (D-13) and it barely misses the 5070
# budget after Whisper + Kokoro.
CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(("llama3.2:3b-instruct-q4_K_M", "llama3.2:3b-instruct", "llama3.2:3b"), 2.5, 100),
    CatalogEntry(("phi3:mini", "phi3"), 2.4, 80),
    CatalogEntry(("gemma2:2b-instruct", "gemma2:2b"), 1.8, 70),
    CatalogEntry(("qwen2.5:1.5b-instruct", "qwen2.5:1.5b"), 1.5, 60),
    CatalogEntry(("llama3.2:1b-instruct-q4_K_M", "llama3.2:1b-instruct", "llama3.2:1b"), 1.1, 50),
)


@dataclass(slots=True)
class OllamaSnapshot:
    status: str  # up | starting | down | missing
    models: list[str] = field(default_factory=list)
    started_by_us: bool = False
    error: str | None = None


_lock = threading.Lock()
_process: subprocess.Popen[bytes] | None = None
_started_by_us = False


def reset_for_tests() -> None:
    """Drop process bookkeeping. Tests that mock Popen must call this."""
    global _process, _started_by_us
    _process = None
    _started_by_us = False


def is_instruct_candidate(name: str) -> bool:
    """False for coder / embed tags so they can never win the pick (D-13)."""
    n = name.lower()
    if any(m in n for m in _CODER_MARKERS):
        return False
    if any(m in n for m in _EMBED_MARKERS):
        return False
    return True


def llm_budget_gb(
    free_vram_gb: float | None,
    available_ram_gb: float,
    use_vram: bool,
    total_ram_gb: float = 0.0,
) -> float:
    """Headroom for the LLM after Whisper + Kokoro are accounted for."""
    if use_vram and free_vram_gb is not None:
        return max(0.0, round(free_vram_gb - WHISPER_KOKORO_VRAM_GB, 2))
    from_total = total_ram_gb - RAM_RESERVE_GB
    from_avail = available_ram_gb - WHISPER_KOKORO_VRAM_GB
    return max(0.0, round(max(from_total, from_avail), 2))


def _base_is_loopback() -> bool:
    host = urlparse(settings.ollama_base_url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


def fetch_tags() -> list[str] | None:
    """Return pulled model names, or None if the daemon did not answer."""
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=TAGS_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — daemon down is the common case
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    names: list[str] = []
    for item in payload.get("models") or []:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            names.append(str(name))
    return names


def snapshot() -> OllamaSnapshot:
    """Cheap liveness read. Never starts a process."""
    tags = fetch_tags()
    if tags is not None:
        return OllamaSnapshot(status="up", models=tags, started_by_us=_started_by_us)

    binary = shutil.which("ollama")
    if binary is None:
        return OllamaSnapshot(status="missing", error="ollama binary is not on PATH")

    if _process is not None and _process.poll() is None:
        return OllamaSnapshot(status="starting", started_by_us=True)

    return OllamaSnapshot(status="down", error=f"Ollama is unreachable at {settings.ollama_base_url}")


def _spawn() -> OllamaSnapshot:
    """Start ``ollama serve`` if we are allowed to. Caller holds ``_lock``."""
    global _process, _started_by_us

    tags = fetch_tags()
    if tags is not None:
        return OllamaSnapshot(status="up", models=tags, started_by_us=_started_by_us)

    if not _base_is_loopback():
        return OllamaSnapshot(
            status="down",
            error=(
                f"Ollama is unreachable at {settings.ollama_base_url} "
                "(not loopback, so this process will not spawn a daemon)."
            ),
        )

    binary = shutil.which("ollama")
    if binary is None:
        return OllamaSnapshot(status="missing", error="ollama binary is not on PATH")

    if _process is not None and _process.poll() is None:
        return OllamaSnapshot(status="starting", started_by_us=True)

    try:
        _process = subprocess.Popen(  # noqa: S603 — binary from PATH, argv is fixed
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        log.warning("Failed to start ollama serve: %s", exc)
        return OllamaSnapshot(status="down", error=f"Could not start ollama serve: {exc}")

    _started_by_us = True
    log.info("Started `ollama serve` (pid %s)", _process.pid)
    return OllamaSnapshot(status="starting", started_by_us=True)


def ensure_ollama(timeout_s: float = ENSURE_WAIT_S) -> OllamaSnapshot:
    """Return a snapshot, starting a local daemon if needed and waiting briefly.

    Safe to call from the health path with ``timeout_s=0`` (spawn, don't wait)
    and from handshake with the default so the first turn does not 404.
    """
    current = snapshot()
    if current.status == "up":
        return current
    if current.status == "missing":
        return current

    with _lock:
        spawned = _spawn()
        if spawned.status in {"up", "missing"}:
            return spawned

    if timeout_s <= 0:
        return spawned

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        tags = fetch_tags()
        if tags is not None:
            return OllamaSnapshot(status="up", models=tags, started_by_us=_started_by_us)
        if _process is not None and _process.poll() is not None:
            return OllamaSnapshot(
                status="down",
                started_by_us=_started_by_us,
                error=f"ollama serve exited with code {_process.returncode}",
            )
        time.sleep(POLL_S)

    return OllamaSnapshot(
        status="starting" if _process is not None and _process.poll() is None else "down",
        started_by_us=_started_by_us,
        error=f"Ollama did not answer /api/tags within {timeout_s:.0f}s",
    )


def _catalog_match(pulled: str) -> CatalogEntry | None:
    n = pulled.lower()
    hits: list[CatalogEntry] = []
    for entry in CATALOG:
        for tag in entry.tags:
            t = tag.lower()
            if n == t or n.startswith(t + "-") or n.startswith(t + "_"):
                hits.append(entry)
                break
    if not hits:
        return None
    return max(hits, key=lambda e: e.rank)


def pick_from_models(
    pulled: Sequence[str],
    budget_gb: float,
    preferred: str | None = None,
) -> tuple[str | None, str]:
    """Choose an instruct model from ``pulled`` that fits ``budget_gb``.

    Returns ``(name, reason)``. ``name`` is None when nothing usable fits;
    the reason is always a human-readable sentence ending in a period.
    """
    usable: list[tuple[str, CatalogEntry]] = []
    skipped = [n for n in pulled if not is_instruct_candidate(n)]
    for name in pulled:
        if not is_instruct_candidate(name):
            continue
        entry = _catalog_match(name)
        if entry is not None:
            usable.append((name, entry))

    pin = preferred if preferred and is_instruct_candidate(preferred) else None
    if pin:
        for name, entry in usable:
            if name.lower() == pin.lower() and entry.vram_gb <= budget_gb:
                return name, (
                    f"Using {name} because it is configured and fits the "
                    f"{budget_gb:.1f} GB LLM budget."
                )

    fits = [(n, e) for n, e in usable if e.vram_gb <= budget_gb]
    if fits:
        fits.sort(key=lambda pair: (-pair[1].rank, -len(pair[0])))
        name, entry = fits[0]
        return name, (
            f"Using {name} — estimated {entry.vram_gb:.1f} GB, "
            f"{budget_gb:.1f} GB LLM headroom after Whisper and Kokoro."
        )

    if usable:
        name, entry = min(usable, key=lambda pair: pair[1].vram_gb)
        return None, (
            f"{name} needs {entry.vram_gb:.1f} GB but only {budget_gb:.1f} GB "
            f"is free after Whisper and Kokoro — routing to Groq."
        )

    if skipped:
        return None, (
            "Ollama is up but no instruct model is pulled "
            f"(found {', '.join(skipped)}) — routing to Groq. "
            f"Pull {PREFERRED_PULL} to run locally."
        )
    if pulled:
        return None, (
            "Ollama is up but none of the pulled models are in the instruct "
            f"catalog ({', '.join(pulled)}) — routing to Groq. "
            f"Pull {PREFERRED_PULL} to run locally."
        )
    return None, (
        f"Ollama is up but no models are pulled — routing to Groq. "
        f"Pull {PREFERRED_PULL} to run locally."
    )
