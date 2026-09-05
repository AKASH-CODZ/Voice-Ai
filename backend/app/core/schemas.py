"""Pydantic models and the WebSocket wire protocol.

The socket carries two kinds of frame:

* **binary** — raw PCM16LE mono. Client→server at ``settings.sample_rate``
  (16 kHz, what Silero and Whisper want); server→client at
  ``AUDIO_OUT_SAMPLE_RATE`` (24 kHz, Kokoro's native rate).
* **text** — a JSON envelope, one of the models below, discriminated by
  its ``type`` field.

Keeping control on the text channel and audio on the binary channel means
we never have to base64 a PCM buffer, which matters when a turn ships a
few hundred kilobytes of speech.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

AUDIO_OUT_SAMPLE_RATE = 24_000


class Mode(StrEnum):
    """The three interaction modes from the spec."""

    CASUAL = "casual"
    TEACHING = "teaching"
    OBSERVATION = "observation"


class EnginePreference(StrEnum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


# ── Client → server ──────────────────────────────────────────


class StartCommand(BaseModel):
    type: Literal["start"] = "start"
    mode: Mode = Mode.CASUAL
    engine: EnginePreference = EnginePreference.AUTO
    topic: str | None = Field(
        default=None,
        description="Optional scenario seed, e.g. 'backend engineer interview'.",
    )
    resume_session_id: str | None = None


class ModeCommand(BaseModel):
    type: Literal["mode"] = "mode"
    mode: Mode


class InterruptCommand(BaseModel):
    """Barge-in: stop speaking immediately and listen."""

    type: Literal["interrupt"] = "interrupt"


class StopCommand(BaseModel):
    type: Literal["stop"] = "stop"


ClientCommand = StartCommand | ModeCommand | InterruptCommand | StopCommand


# ── Server → client ──────────────────────────────────────────


class ReadyEvent(BaseModel):
    type: Literal["ready"] = "ready"
    session_id: str
    mode: Mode
    engine: str
    engine_reason: str
    audio_in_sample_rate: int
    audio_out_sample_rate: int = AUDIO_OUT_SAMPLE_RATE
    frame_samples: int
    hardware: dict[str, Any]


class VadEvent(BaseModel):
    type: Literal["vad"] = "vad"
    speaking: bool


class PartialTranscriptEvent(BaseModel):
    type: Literal["partial_transcript"] = "partial_transcript"
    text: str


class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    turn_id: int
    role: Role
    text: str
    hesitation: bool = False
    correction_made: bool = False


class AgentDeltaEvent(BaseModel):
    """A streamed LLM token chunk, emitted before TTS audio for that chunk."""

    type: Literal["agent_delta"] = "agent_delta"
    text: str


class SpeakingEvent(BaseModel):
    type: Literal["speaking"] = "speaking"
    active: bool


class StallEvent(BaseModel):
    """Teaching mode fired the hesitation watchdog."""

    type: Literal["stall"] = "stall"
    silence_ms: int


class LatencyEvent(BaseModel):
    type: Literal["latency"] = "latency"
    turn_id: int
    stt_ms: float
    llm_ttft_ms: float
    llm_ms: float
    tts_ms: float
    e2e_ms: float
    engine: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    fatal: bool = False


class EngineSwitchedEvent(BaseModel):
    """Emitted when the router degrades local→cloud mid-session."""

    type: Literal["engine_switched"] = "engine_switched"
    engine: str
    reason: str


ServerEvent = (
    ReadyEvent
    | VadEvent
    | PartialTranscriptEvent
    | TranscriptEvent
    | AgentDeltaEvent
    | SpeakingEvent
    | StallEvent
    | LatencyEvent
    | ErrorEvent
    | EngineSwitchedEvent
)


# ── REST payloads ────────────────────────────────────────────


class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    mode: Mode
    topic: str | None
    engine: str
    started_at: str
    ended_at: str | None
    turn_count: int


class TurnRecord(BaseModel):
    turn_id: int
    session_id: str
    role: Role
    text: str
    timestamp: str
    hesitation: bool
    correction_made: bool
    stt_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    e2e_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    engine_mode: str
    recommended_engine: str
    reason: str
    hardware: dict[str, Any]
