"""Abstract engine interfaces.

The whole local-vs-cloud story rests on these three protocols. The
orchestrator never learns whether it is talking to a 5070 or to Groq — it holds
a :class:`VoiceEngine` and calls the same four methods either way. That is what
makes the hardware router a one-line swap instead of a branching mess through
the pipeline.

Two interface choices are load-bearing:

* ``LLMEngine.stream`` is an **async iterator of token deltas**, not a coroutine
  returning a string. A blocking call here would defeat the sentence chunker
  and put us straight back at "wait for the whole paragraph".
* ``TTSEngine.synthesize`` is likewise an **async iterator of PCM chunks**, so a
  long sentence starts playing before it has finished rendering.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transcript:
    text: str
    language: str = "en"
    duration_ms: float = 0.0
    latency_ms: float = 0.0
    # Whisper's own confidence; low values are a decent proxy for mumbling.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class STTEngine(abc.ABC):
    """Speech → text."""

    name: str = "stt"

    @abc.abstractmethod
    async def load(self) -> None: ...

    @abc.abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript: ...

    async def unload(self) -> None:
        return None


class LLMEngine(abc.ABC):
    """Chat completion, streamed."""

    name: str = "llm"

    @abc.abstractmethod
    async def load(self) -> None: ...

    @abc.abstractmethod
    def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield token deltas as they arrive. Must be cancellable mid-stream:
        barge-in works by cancelling the task that drives this iterator."""

    async def unload(self) -> None:
        return None


class TTSEngine(abc.ABC):
    """Text → speech."""

    name: str = "tts"
    sample_rate: int = 24_000

    @abc.abstractmethod
    async def load(self) -> None: ...

    @abc.abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield float32 mono chunks at ``self.sample_rate``."""

    async def unload(self) -> None:
        return None


@dataclass(slots=True)
class VoiceEngine:
    """One complete STT → LLM → TTS loadout, plus a label for the UI badge."""

    kind: str            # "local" | "cloud"
    label: str           # e.g. "Local Engine (RTX 5070)"
    stt: STTEngine
    llm: LLMEngine
    tts: TTSEngine

    async def load(self) -> None:
        # Sequential, not gathered: on an 8 GB card, loading Whisper and Kokoro
        # concurrently can spike allocation past the ceiling and OOM even
        # though the steady-state footprint fits comfortably.
        await self.stt.load()
        await self.llm.load()
        await self.tts.load()

    async def unload(self) -> None:
        for engine in (self.tts, self.llm, self.stt):
            try:
                await engine.unload()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
