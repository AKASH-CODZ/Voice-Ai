"""Barge-in: cancel the speaking turn without a live microphone.

Live-mic confirmation is still a human checklist item. These tests lock the
invariants that make a live test even possible: five loud frames while the
agent is speaking cancel the turn; a quiet frame resets the counter; the
segmenter is not fed while the agent talks (that is the self-echo loop).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np
import pytest

from app.core.config import settings
from app.core.schemas import Mode, SpeakingEvent
from app.engines.base import LLMEngine, STTEngine, TTSEngine, Transcript, VoiceEngine
from app.pipeline.orchestrator import BARGE_IN_FRAMES, VoiceOrchestrator
from app.pipeline.state import ConversationState
from app.pipeline.vad import UtteranceSegmenter


class FakeVAD:
    def __init__(self, prob: float = 0.0) -> None:
        self.prob = prob

    def __call__(self, frame: np.ndarray) -> float:
        return self.prob

    def reset(self) -> None:
        return None


class FakeSTT(STTEngine):
    async def load(self) -> None:
        return None

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        return Transcript(text="I want to practice for an interview.", latency_ms=1.0)


class SlowLLM(LLMEngine):
    """Holds the turn open so barge-in has something to cancel."""

    def __init__(self, hold: asyncio.Event) -> None:
        self.hold = hold

    async def load(self) -> None:
        return None

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        yield "Hello there. "
        await self.hold.wait()
        yield "This should never be spoken."


class FakeTTS(TTSEngine):
    async def load(self) -> None:
        return None

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        yield np.zeros(512, dtype=np.float32)


class NullTelemetry:
    def record(self, event) -> None:  # noqa: ANN001
        return None


def _orch(vad: FakeVAD, hold: asyncio.Event) -> tuple[VoiceOrchestrator, list]:
    events: list = []

    async def send_event(ev) -> None:  # noqa: ANN001
        events.append(ev)

    async def send_audio(_pcm: bytes) -> None:
        return None

    engine = VoiceEngine(
        kind="local",
        label="test",
        stt=FakeSTT(),
        llm=SlowLLM(hold),
        tts=FakeTTS(),
    )
    orch = VoiceOrchestrator(
        state=ConversationState(session_id="barge", mode=Mode.CASUAL),
        engine=engine,
        segmenter=UtteranceSegmenter(vad=vad),
        telemetry=NullTelemetry(),  # type: ignore[arg-type]
        send_event=send_event,
        send_audio=send_audio,
    )
    return orch, events


def _frame() -> np.ndarray:
    return np.zeros(settings.frame_samples, dtype=np.float32)


async def _wait_speaking(orch: VoiceOrchestrator) -> None:
    for _ in range(200):
        if orch._speaking:
            orch._barge_in_grace_until = 0.0
            return
        await asyncio.sleep(0.01)
    raise AssertionError("turn never started speaking")


@pytest.mark.asyncio
async def test_barge_in_grace_ignores_echo_of_opening_words():
    """Speaker echo of the first syllables must not cut the reply."""
    hold = asyncio.Event()
    vad = FakeVAD(0.99)
    orch, _ = _orch(vad, hold)
    await orch._begin_turn(np.zeros(settings.sample_rate, dtype=np.float32))
    for _ in range(200):
        if orch._speaking:
            break
        await asyncio.sleep(0.01)
    orch._barge_in_grace_until = time.monotonic() + 10

    for _ in range(BARGE_IN_FRAMES * 3):
        await orch._on_frame(_frame())

    assert orch._speaking is True
    hold.set()
    await orch.close()


@pytest.mark.asyncio
async def test_five_loud_frames_cancel_the_turn():
    hold = asyncio.Event()
    vad = FakeVAD(0.99)
    orch, events = _orch(vad, hold)
    await orch._begin_turn(np.zeros(settings.sample_rate, dtype=np.float32))
    await _wait_speaking(orch)
    assert orch._turn_task is not None and not orch._turn_task.done()

    for _ in range(BARGE_IN_FRAMES):
        await orch._on_frame(_frame())

    await asyncio.sleep(0.05)
    assert orch._speaking is False
    assert orch._turn_task.done()
    assert any(isinstance(e, SpeakingEvent) and e.active is False for e in events)
    await orch.close()


@pytest.mark.asyncio
async def test_quiet_frame_resets_barge_in_counter():
    hold = asyncio.Event()
    vad = FakeVAD(0.99)
    orch, _ = _orch(vad, hold)
    await orch._begin_turn(np.zeros(settings.sample_rate, dtype=np.float32))
    await _wait_speaking(orch)

    for _ in range(BARGE_IN_FRAMES - 1):
        await orch._on_frame(_frame())
    vad.prob = 0.1
    await orch._on_frame(_frame())
    vad.prob = 0.99
    for _ in range(BARGE_IN_FRAMES - 1):
        await orch._on_frame(_frame())

    assert orch._speaking is True, "counter must reset after a quiet frame"
    hold.set()
    await orch.close()


@pytest.mark.asyncio
async def test_disabled_barge_in_does_not_interrupt(monkeypatch: pytest.MonkeyPatch):
    from app.core.config import settings as s

    monkeypatch.setattr(s, "barge_in_enabled", False)
    hold = asyncio.Event()
    vad = FakeVAD(0.99)
    orch, _ = _orch(vad, hold)
    await orch._begin_turn(np.zeros(settings.sample_rate, dtype=np.float32))
    await _wait_speaking(orch)

    for _ in range(BARGE_IN_FRAMES * 3):
        await orch._on_frame(_frame())

    assert orch._speaking is True
    hold.set()
    await orch.close()


@pytest.mark.asyncio
async def test_speaking_does_not_feed_the_segmenter():
    """The self-echo loop: agent audio must never become a user utterance."""
    hold = asyncio.Event()
    vad = FakeVAD(0.99)
    orch, _ = _orch(vad, hold)
    await orch._begin_turn(np.zeros(settings.sample_rate, dtype=np.float32))
    await _wait_speaking(orch)

    pushed = []
    original = orch.segmenter.push

    def spy(frame: np.ndarray):
        pushed.append(frame)
        return original(frame)

    orch.segmenter.push = spy  # type: ignore[method-assign]
    # Four loud frames — one short of barge-in — so the turn stays speaking.
    for _ in range(BARGE_IN_FRAMES - 1):
        await orch._on_frame(_frame())

    assert pushed == [], "segmenter must not see frames while the agent is talking"
    hold.set()
    await orch.close()


@pytest.mark.asyncio
async def test_interrupt_hands_the_floor_back():
    hold = asyncio.Event()
    orch, events = _orch(FakeVAD(0.0), hold)
    await orch._begin_turn(np.zeros(settings.sample_rate, dtype=np.float32))
    await _wait_speaking(orch)
    await orch.interrupt()
    assert orch._speaking is False
    assert any(isinstance(e, SpeakingEvent) and e.active is False for e in events)
    await orch.close()
