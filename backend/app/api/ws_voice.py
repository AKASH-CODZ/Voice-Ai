"""The real-time voice WebSocket.

Protocol
--------
Client → server
  * **binary** — PCM16LE mono @ ``SAMPLE_RATE`` (16 kHz).
  * **text**   — one JSON command: ``start`` | ``mode`` | ``interrupt`` | ``stop``.

Server → client
  * **binary** — PCM16LE mono @ 24 kHz (TTS output).
  * **text**   — JSON events (see ``core/schemas.py``).

The connection is single-reader: this handler owns ``receive()``. Everything
that sends does so through ``_send``/``_send_audio``, which serialise on a lock
— Starlette's WebSocket is not safe against concurrent sends, and the turn task
and the control loop genuinely do write at the same time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.deps import ws_client_ip
from app.core.config import settings
from app.core.database import db
from app.core.schemas import (
    AUDIO_OUT_SAMPLE_RATE,
    EnginePreference,
    ErrorEvent,
    Mode,
    ReadyEvent,
    ServerEvent,
    StartCommand,
)
from app.engines.router import router as engine_router
from app.pipeline.orchestrator import VoiceOrchestrator
from app.pipeline.state import ConversationState
from app.pipeline.telemetry import TelemetryWriter
from app.pipeline.vad import SileroVAD, UtteranceSegmenter

log = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

# The VAD model is ~1.8 MB and stateless between sessions once reset, but each
# connection needs its own LSTM state, so we share nothing and build per socket.
# Loading is fast (<50 ms) so this is not worth pooling.

START_TIMEOUT_S = 30.0


@router.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()

    async def _send(event: ServerEvent) -> None:
        async with send_lock:
            await websocket.send_text(event.model_dump_json())

    async def _send_audio(pcm: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(pcm)

    ip = ws_client_ip(websocket)
    orchestrator: VoiceOrchestrator | None = None
    telemetry: TelemetryWriter | None = None
    session_id: str | None = None

    try:
        # ── handshake ────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=START_TIMEOUT_S)
            start = StartCommand.model_validate_json(raw)
        except TimeoutError:
            await _send(ErrorEvent(message="Timed out waiting for a start command.", fatal=True))
            await websocket.close(code=1008)
            return
        except (ValidationError, json.JSONDecodeError) as exc:
            await _send(ErrorEvent(message=f"Malformed start command: {exc}", fatal=True))
            await websocket.close(code=1003)
            return

        user_id = await db.upsert_user(ip)

        # ── engine selection ─────────────────────────────────
        try:
            engine, report, reason = await engine_router.acquire(start.engine)
        except Exception as exc:  # noqa: BLE001
            await _send(ErrorEvent(message=f"No usable engine. {exc}", fatal=True))
            await websocket.close(code=1011)
            return

        # ── session: resume or create ────────────────────────
        state = ConversationState(session_id="", mode=start.mode, topic=start.topic,
                                  engine=engine.kind)

        resumed = None
        if start.resume_session_id:
            resumed = await db.get_session(start.resume_session_id)
            if resumed is not None and resumed.user_id != user_id:
                resumed = None  # never hand one IP another IP's transcript

        if resumed is not None:
            session_id = resumed.session_id
            state.session_id = session_id
            state.topic = resumed.topic or start.topic
            await db.update_session_engine(session_id, engine.kind)
            await db.update_session_mode(session_id, start.mode)
            state.hydrate(await db.recent_turns(session_id, settings.context_window_turns * 2))
        else:
            session_id = await db.create_session(user_id, start.mode, engine.kind, start.topic)
            state.session_id = session_id

        # ── pipeline assembly ────────────────────────────────
        try:
            segmenter = UtteranceSegmenter(SileroVAD())
        except FileNotFoundError as exc:
            await _send(ErrorEvent(message=str(exc), fatal=True))
            await websocket.close(code=1011)
            return

        telemetry = TelemetryWriter(db)
        await telemetry.start()

        orchestrator = VoiceOrchestrator(
            state=state,
            engine=engine,
            segmenter=segmenter,
            telemetry=telemetry,
            send_event=_send,
            send_audio=_send_audio,
        )

        await _send(ReadyEvent(
            session_id=session_id,
            mode=state.mode,
            engine=engine.kind,
            engine_reason=reason,
            audio_in_sample_rate=settings.sample_rate,
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
            frame_samples=settings.frame_samples,
            hardware=report.to_dict(),
        ))
        log.info(
            "[%s] session open — user=%s ip=%s mode=%s engine=%s (%s)",
            session_id, user_id, ip, state.mode.value, engine.kind, reason,
        )

        # The agent speaks first: an open mic with no prompt is the most
        # common way a voice demo stalls before it starts.
        await orchestrator.greet()

        # ── main loop ────────────────────────────────────────
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if (payload := message.get("bytes")) is not None:
                await orchestrator.push_audio(payload)
                continue

            text = message.get("text")
            if not text:
                continue

            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                continue

            kind = command.get("type")

            if kind == "mode":
                try:
                    mode = Mode(command["mode"])
                except (KeyError, ValueError):
                    await _send(ErrorEvent(message=f"Unknown mode: {command.get('mode')!r}"))
                    continue
                await orchestrator.set_mode(mode)
                await db.update_session_mode(session_id, mode)

            elif kind == "interrupt":
                await orchestrator.interrupt()

            elif kind == "stop":
                break

            elif kind == "engine":
                # Manual override from the badge, applied between turns.
                try:
                    preference = EnginePreference(command.get("engine", "auto"))
                except ValueError:
                    continue
                new_engine, _, new_reason = await engine_router.acquire(preference)
                if new_engine.kind != orchestrator.engine.kind:
                    await orchestrator.switch_engine(new_engine, new_reason)
                    await db.update_session_engine(session_id, new_engine.kind)

    except WebSocketDisconnect:
        log.info("[%s] client disconnected", session_id or "-")
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] socket error: %s", session_id or "-", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await _send(ErrorEvent(message=f"{type(exc).__name__}: {exc}", fatal=True))
    finally:
        if orchestrator is not None:
            await orchestrator.close()
        if telemetry is not None:
            # Drain the backlog before the connection dies — the spec wants
            # logging 100% of the time, which includes the last turn.
            await telemetry.stop()
        if session_id is not None:
            with contextlib.suppress(Exception):
                await db.end_session(session_id)
        with contextlib.suppress(Exception):
            await websocket.close()
        log.info("[%s] session closed", session_id or "-")
