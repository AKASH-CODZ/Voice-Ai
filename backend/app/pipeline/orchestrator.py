"""The turn loop: audio in → VAD → STT → LLM → TTS → audio out.

This is where the latency budget is actually spent, so a few structural
choices are worth stating up front.

**One task per turn, and it is cancellable.** The entire STT→LLM→TTS chain for
a turn runs inside ``self._turn_task``. Barge-in is therefore not a special
protocol — it is ``task.cancel()``. Every stage cooperates by being an async
generator, so cancellation unwinds the LLM stream, the TTS generator and the
ffmpeg subprocess together, with no orphaned work still writing audio into a
socket the user has already talked over.

**The microphone is gated while the AI speaks.** Browser-side AEC removes most
of the echo, but "most" is not enough: one leaked syllable transcribed as user
speech starts a feedback loop where the agent answers itself forever. So while
the AI is speaking we stop feeding the segmenter entirely and run a much
stricter barge-in detector instead — a high threshold sustained across several
frames. That asymmetry is deliberate: a false barge-in costs an interrupted
sentence, while a false transcription costs the entire conversation.

**Latency is measured from end-of-utterance, not from turn start.** The number
that matters to a user is the gap between them finishing a sentence and hearing
the first syllable back. That is ``e2e_ms``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

import numpy as np

from app.core.audio import float32_to_pcm16, pcm16_to_float32
from app.core.config import settings
from app.core.schemas import (
    AgentDeltaEvent,
    EngineSwitchedEvent,
    ErrorEvent,
    LatencyEvent,
    Mode,
    Role,
    ServerEvent,
    SpeakingEvent,
    StallEvent,
    TranscriptEvent,
    VadEvent,
)
from app.engines.base import VoiceEngine
from app.pipeline.chunker import SentenceChunker
from app.pipeline.state import ConversationState
from app.pipeline.telemetry import TelemetryWriter, TurnEvent
from app.pipeline.vad import UtteranceSegmenter

log = logging.getLogger(__name__)

SendEvent = Callable[[ServerEvent], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

# Barge-in must be much harder to trigger than ordinary speech onset: these
# frames are being captured while the speaker is actively playing audio.
BARGE_IN_THRESHOLD = 0.85
BARGE_IN_FRAMES = 5           # ~160 ms of confident speech at 32 ms/frame
# Ignore mic frames for this long after the first TTS byte. Speaker echo of
# the opening words otherwise looks like a barge-in and cuts the reply.
BARGE_IN_GRACE_S = 0.6

# Whisper emits these for silence, breath and clicks. Speaking them back is
# worse than saying nothing.
_JUNK_TRANSCRIPTS = {
    "", ".", "..", "...", "you", "thank you.", "thanks for watching!", "bye.",
    "thank you for watching!", "[blank_audio]", "(silence)", "um", "uh",
}


class VoiceOrchestrator:
    """Drives one conversation over one WebSocket connection."""

    def __init__(
        self,
        *,
        state: ConversationState,
        engine: VoiceEngine,
        segmenter: UtteranceSegmenter,
        telemetry: TelemetryWriter,
        send_event: SendEvent,
        send_audio: SendAudio,
    ) -> None:
        self.state = state
        self.engine = engine
        self.segmenter = segmenter
        self.telemetry = telemetry
        self._send_event = send_event
        self._send_audio = send_audio

        self._frame_bytes = settings.frame_bytes
        self._residual = bytearray()

        self._speaking = False           # AI currently rendering audio
        self._turn_task: asyncio.Task[None] | None = None
        self._barge_in_run = 0
        self._barge_in_grace_until = 0.0
        self._closed = False

        # Guards against two utterances racing into overlapping turns when the
        # user speaks again while STT is still running on the previous one.
        self._turn_lock = asyncio.Lock()

    # ── inbound audio ────────────────────────────────────────

    async def push_audio(self, pcm: bytes) -> None:
        """Consume raw PCM16LE from the socket, aligned into VAD frames.

        The browser sends whatever its AudioWorklet buffer produced, which is
        not guaranteed to be a whole number of 512-sample frames. We carry the
        remainder forward rather than zero-padding — padding would inject a
        fake silence frame into the middle of a word every buffer boundary.
        """
        if self._closed:
            return

        self._residual.extend(pcm)
        while len(self._residual) >= self._frame_bytes:
            raw = bytes(self._residual[: self._frame_bytes])
            del self._residual[: self._frame_bytes]
            await self._on_frame(pcm16_to_float32(raw))

    async def _on_frame(self, frame: np.ndarray) -> None:
        if self._speaking:
            await self._check_barge_in(frame)
            return

        decision = self.segmenter.push(frame)

        if decision.speech_started:
            await self._send_event(VadEvent(speaking=True))

        if decision.stalled_ms is not None:
            await self._on_stall(decision.stalled_ms)

        if decision.utterance is not None:
            await self._send_event(VadEvent(speaking=False))
            await self._begin_turn(decision.utterance)

    async def _check_barge_in(self, frame: np.ndarray) -> None:
        if not settings.barge_in_enabled:
            return
        if time.monotonic() < self._barge_in_grace_until:
            return
        prob = self.segmenter.vad(frame)
        if prob >= BARGE_IN_THRESHOLD:
            self._barge_in_run += 1
            if self._barge_in_run >= BARGE_IN_FRAMES:
                log.info("[%s] barge-in detected", self.state.session_id)
                self._barge_in_run = 0
                await self.interrupt()
        else:
            self._barge_in_run = 0

    # ── stall watchdog ───────────────────────────────────────

    async def _on_stall(self, silence_ms: int) -> None:
        should_intervene = self.state.note_stall(silence_ms)
        await self._send_event(StallEvent(silence_ms=silence_ms))
        log.info(
            "[%s] stall %d ms (mode=%s, intervene=%s)",
            self.state.session_id, silence_ms, self.state.mode.value, should_intervene,
        )
        if should_intervene:
            # Teaching mode only: run a turn with no user text — the state
            # machine has already queued the [USER_STALLED] system notice.
            await self._begin_turn(None)

    # ── turn execution ───────────────────────────────────────

    async def _begin_turn(self, audio: np.ndarray | None) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._turn_task = asyncio.create_task(
            self._run_turn(audio), name=f"turn-{self.state.turn_index}"
        )

    async def _run_turn(self, audio: np.ndarray | None) -> None:
        utterance_end = time.perf_counter()
        stt_ms = llm_ttft_ms = llm_ms = tts_ms = e2e_ms = 0.0
        first_audio_sent = False

        try:
            async with self._turn_lock:
                user_text: str | None = None

                # ── STT ──
                if audio is not None:
                    result = await self.engine.stt.transcribe(audio, settings.sample_rate)
                    stt_ms = result.latency_ms
                    user_text = result.text.strip()

                    if _is_junk(user_text, result.no_speech_prob):
                        log.debug("[%s] discarded junk transcript %r",
                                  self.state.session_id, user_text)
                        self.segmenter.arm_stall_watchdog()
                        return

                    self.state.append_user(user_text)
                    turn_id = self.state.turn_index
                    await self._send_event(
                        TranscriptEvent(
                            turn_id=turn_id,
                            role=Role.USER,
                            text=user_text,
                            hesitation=self.state.last_turn_hesitated,
                        )
                    )
                    self.telemetry.record(
                        TurnEvent(
                            session_id=self.state.session_id,
                            role=Role.USER,
                            text=user_text,
                            hesitation=self.state.last_turn_hesitated,
                            stt_ms=stt_ms,
                        )
                    )

                # ── LLM → chunker → TTS ──
                messages = self.state.build_messages(user_text)
                chunker = SentenceChunker()
                spoken: list[str] = []

                self._speaking = True
                self._barge_in_run = 0
                self._barge_in_grace_until = 0.0
                self.segmenter.disarm_stall_watchdog()
                await self._send_event(SpeakingEvent(active=True))

                llm_started = time.perf_counter()
                got_first_token = False

                async for delta in self.engine.llm.stream(messages):
                    if not got_first_token:
                        llm_ttft_ms = (time.perf_counter() - llm_started) * 1000.0
                        got_first_token = True

                    await self._send_event(AgentDeltaEvent(text=delta))

                    for chunk in chunker.push(delta):
                        spoken.append(chunk)
                        tts_started = time.perf_counter()
                        async for pcm_chunk in self.engine.tts.synthesize(chunk):
                            if not first_audio_sent:
                                tts_ms = (time.perf_counter() - tts_started) * 1000.0
                                e2e_ms = (time.perf_counter() - utterance_end) * 1000.0
                                first_audio_sent = True
                                self._barge_in_grace_until = time.monotonic() + BARGE_IN_GRACE_S
                            await self._send_audio(float32_to_pcm16(pcm_chunk))

                llm_ms = (time.perf_counter() - llm_started) * 1000.0

                # Drain whatever the model left without terminal punctuation.
                tail = chunker.flush()
                if tail:
                    spoken.append(tail)
                    tts_started = time.perf_counter()
                    async for pcm_chunk in self.engine.tts.synthesize(tail):
                        if not first_audio_sent:
                            tts_ms = (time.perf_counter() - tts_started) * 1000.0
                            e2e_ms = (time.perf_counter() - utterance_end) * 1000.0
                            first_audio_sent = True
                            self._barge_in_grace_until = time.monotonic() + BARGE_IN_GRACE_S
                        await self._send_audio(float32_to_pcm16(pcm_chunk))

                # ── commit the turn ──
                reply = " ".join(spoken).strip()
                if reply:
                    self.state.append_assistant(reply)
                    await self._send_event(
                        TranscriptEvent(
                            turn_id=self.state.turn_index,
                            role=Role.ASSISTANT,
                            text=reply,
                            correction_made=self.state.last_turn_corrected,
                        )
                    )
                    self.telemetry.record(
                        TurnEvent(
                            session_id=self.state.session_id,
                            role=Role.ASSISTANT,
                            text=reply,
                            correction_made=self.state.last_turn_corrected,
                            llm_ms=llm_ms,
                            tts_ms=tts_ms,
                            e2e_ms=e2e_ms,
                        )
                    )
                    await self._send_event(
                        LatencyEvent(
                            turn_id=self.state.turn_index,
                            stt_ms=round(stt_ms, 1),
                            llm_ttft_ms=round(llm_ttft_ms, 1),
                            llm_ms=round(llm_ms, 1),
                            tts_ms=round(tts_ms, 1),
                            e2e_ms=round(e2e_ms, 1),
                            engine=self.engine.kind,
                        )
                    )
                    log.info(
                        "[%s] turn %d — stt %.0f | ttft %.0f | llm %.0f | tts %.0f | e2e %.0f ms",
                        self.state.session_id, self.state.turn_index,
                        stt_ms, llm_ttft_ms, llm_ms, tts_ms, e2e_ms,
                    )

        except asyncio.CancelledError:
            log.debug("[%s] turn cancelled (barge-in or shutdown)", self.state.session_id)
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] turn failed: %s", self.state.session_id, exc, exc_info=True)
            await self._send_event(
                ErrorEvent(message=f"Turn failed: {type(exc).__name__}: {exc}")
            )
        finally:
            self._speaking = False
            self.state.clear_turn_flags()
            with contextlib.suppress(Exception):
                await self._send_event(SpeakingEvent(active=False))
            # Only now does user silence mean "stuck" rather than "listening".
            self.segmenter.arm_stall_watchdog()

    # ── control ──────────────────────────────────────────────

    async def interrupt(self) -> None:
        """Stop speaking immediately and hand the floor back."""
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._speaking = False
        self._barge_in_grace_until = 0.0
        self.segmenter.reset()
        self.segmenter.arm_stall_watchdog()
        with contextlib.suppress(Exception):
            await self._send_event(SpeakingEvent(active=False))

    async def set_mode(self, mode: Mode) -> None:
        self.state.set_mode(mode)

    async def switch_engine(self, engine: VoiceEngine, reason: str) -> None:
        self.engine = engine
        self.state.engine = engine.kind
        await self._send_event(EngineSwitchedEvent(engine=engine.kind, reason=reason))

    async def greet(self) -> None:
        """Speak first so the user never faces a silent microphone."""
        await self._begin_turn(None)

    async def close(self) -> None:
        self._closed = True
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        self._residual.clear()


def _is_junk(text: str, no_speech_prob: float | None) -> bool:
    """Filter Whisper's silence hallucinations before they reach the LLM."""
    if not text:
        return True
    if no_speech_prob is not None and no_speech_prob > 0.75:
        return True
    normalized = text.strip().lower()
    if normalized in _JUNK_TRANSCRIPTS:
        return True
    # A single stray token with no letters is punctuation noise.
    return not any(ch.isalpha() for ch in normalized)
