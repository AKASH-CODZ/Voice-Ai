"""Background telemetry writer — 100% logging that never blocks audio.

The plan's requirement is that logging runs "100% of the time in the
background". The failure mode it is guarding against is real: an `await` on a
database commit inside the audio callback adds jitter to every frame, and
jitter in a voice pipeline is audible as choppiness.

So nothing here is awaited by the caller. ``record()`` is a non-blocking put
onto an unbounded queue; a single consumer task drains it and does the actual
SQLite writes. If the queue somehow grows without bound we drop the *oldest*
telemetry rather than applying backpressure to the conversation — a missing
latency sample is a far better outcome than a stuttering agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from app.core.database import Database
from app.core.schemas import Role

log = logging.getLogger(__name__)

MAX_QUEUE = 2048


@dataclass(slots=True)
class TurnEvent:
    session_id: str
    role: Role
    text: str
    hesitation: bool = False
    correction_made: bool = False
    stt_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    e2e_ms: float | None = None


class TelemetryWriter:
    """Fire-and-forget turn logger backed by one drain task."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0
        # Populated by the drain task so the orchestrator can echo the real
        # row id back to the client without ever awaiting the write.
        self.last_turn_id: int = 0

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain(), name="telemetry-drain")
            log.debug("Telemetry writer started")

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)          # sentinel: finish the backlog, then exit
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(self._task, timeout=5.0)
        self._task = None
        if self._dropped:
            log.warning("Telemetry dropped %d event(s) under load", self._dropped)

    def record(self, event: TurnEvent) -> None:
        """Non-blocking. Safe to call from anywhere on the event loop."""
        if self._queue.qsize() >= MAX_QUEUE:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()      # shed the oldest, keep the newest
                self._dropped += 1
        self._queue.put_nowait(event)

    async def _drain(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            try:
                self.last_turn_id = await self._db.add_turn(
                    event.session_id,
                    event.role,
                    event.text,
                    hesitation=event.hesitation,
                    correction_made=event.correction_made,
                    stt_ms=event.stt_ms,
                    llm_ms=event.llm_ms,
                    tts_ms=event.tts_ms,
                    e2e_ms=event.e2e_ms,
                )
            except Exception as exc:  # noqa: BLE001
                # A logging failure must never take down a live conversation.
                log.error("Telemetry write failed: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()
