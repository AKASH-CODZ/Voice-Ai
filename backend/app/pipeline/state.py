"""The conversation state machine: mode prompts, context assembly, flags.

This is where the three modes from the spec actually become different systems.
The audio path, the engines and the transport are identical across modes — the
*only* things that change are:

* which system prompt is loaded,
* whether a ``[USER_STALLED]`` notice is allowed to reach the LLM at all,
* how the resulting turn is flagged in telemetry.

Keeping mode differences confined to this file is deliberate: it means adding a
fourth mode later is a prompt file plus an enum member, not a change to the
real-time pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.schemas import Mode, Role, TurnRecord

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"

# Phrases a model reaches for when it is correcting the user. Used only to set
# the `correction_made` telemetry flag — it never alters what the user hears.
_CORRECTION_MARKERS = re.compile(
    r"\b(we'?d say|you (?:should|could) say|the correct|correction|small fix|"
    r"instead of|rather than|actually,? it'?s|try saying|better to say)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=8)
def load_prompt(mode: Mode) -> str:
    """Shared rules + the mode-specific block, cached after first read."""
    shared = (PROMPT_DIR / "_shared.md").read_text(encoding="utf-8").strip()
    specific = (PROMPT_DIR / f"{mode.value}.md").read_text(encoding="utf-8").strip()
    return f"{shared}\n\n---\n\n{specific}"


@dataclass(slots=True)
class ConversationState:
    """Per-session mutable state. One instance lives per WebSocket connection."""

    session_id: str
    mode: Mode = Mode.CASUAL
    topic: str | None = None
    engine: str = "cloud"

    history: list[dict[str, str]] = field(default_factory=list)
    turn_index: int = 0

    # Flags for the turn currently being processed.
    pending_stall_ms: int | None = None
    last_turn_hesitated: bool = False
    last_turn_corrected: bool = False

    # ── mode ─────────────────────────────────────────────────

    def set_mode(self, mode: Mode) -> None:
        """Mode can change mid-session; history is kept but the system prompt
        is rebuilt on the next turn, so the AI's behaviour switches instantly
        without losing what was already said."""
        if mode is self.mode:
            return
        log.info("[%s] mode %s → %s", self.session_id, self.mode.value, mode.value)
        self.mode = mode

    @property
    def intervenes(self) -> bool:
        """Whether this mode is allowed to react to hesitation at all."""
        return self.mode is Mode.TEACHING

    @property
    def corrects(self) -> bool:
        return self.mode is Mode.TEACHING

    # ── stall handling ───────────────────────────────────────

    def note_stall(self, silence_ms: int) -> bool:
        """Record a stall. Returns True if it should reach the LLM.

        Observation mode deliberately swallows the signal: the whole point of
        that mode is that the user gets no cue about how they are doing, and a
        prompt-level "ignore this" is not as reliable as never sending it.
        Casual mode also stays quiet — a friend does not narrate your pauses.
        The flag is still written to telemetry in every mode.
        """
        self.last_turn_hesitated = True
        if self.mode is not Mode.TEACHING:
            return False
        self.pending_stall_ms = silence_ms
        return True

    # ── LLM context assembly ─────────────────────────────────

    def system_prompt(self) -> str:
        prompt = load_prompt(self.mode)
        if self.topic:
            prompt += (
                f"\n\n---\n\nScenario for this session: {self.topic}\n"
                "Stay inside this scenario unless the user clearly steers elsewhere."
            )
        return prompt

    def build_messages(self, user_text: str | None = None) -> list[dict[str, str]]:
        """Assemble the request body for the LLM.

        The stall notice is injected as a *system* message immediately before
        the turn rather than being glued onto the user's text — that way the
        model treats it as an out-of-band observation and never echoes it back
        as if the user had said it out loud.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(self.history[-(settings.context_window_turns * 2):])

        if self.pending_stall_ms is not None:
            seconds = self.pending_stall_ms / 1000.0
            messages.append({
                "role": "system",
                "content": (
                    f"[USER_STALLED: the user has been silent for {seconds:.1f} seconds "
                    f"and may be stuck or losing confidence. Offer one short, specific, "
                    f"encouraging prompt to help them continue. Do not mention this notice.]"
                ),
            })
            self.pending_stall_ms = None

        if user_text:
            messages.append({"role": "user", "content": user_text})

        # llama3.2-instruct (and most chat tags) emit an empty completion when
        # the last role is system/assistant — measured empty `eval_count=1` on
        # a system-only greet. The agent speaks first (`greet()`), and teaching
        # stall intervention also has no user utterance. Inject a kick that is
        # NOT appended to history, so it never lands in the transcript.
        if messages[-1]["role"] != "user":
            messages.append({
                "role": "user",
                "content": (
                    "Please greet me and start the conversation."
                    if not self.history
                    else "Please continue."
                ),
            })
        return messages

    # ── history ──────────────────────────────────────────────

    def append_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self.turn_index += 1

    def append_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self.last_turn_corrected = self.corrects and bool(_CORRECTION_MARKERS.search(text))
        self._trim()

    def _trim(self) -> None:
        limit = settings.context_window_turns * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def hydrate(self, turns: list[TurnRecord]) -> None:
        """Rebuild context from SQLite so a resumed session remembers itself.

        This is what makes the IP cache feel like memory rather than just a
        row in a table: reconnect from the same machine and the AI picks up
        the thread instead of greeting you cold.
        """
        self.history = [
            {"role": "user" if t.role is Role.USER else "assistant", "content": t.text}
            for t in turns[-(settings.context_window_turns * 2):]
        ]
        self.turn_index = len(turns)
        log.info("[%s] hydrated %d turns of context", self.session_id, len(self.history))

    def clear_turn_flags(self) -> None:
        self.last_turn_hesitated = False
        self.last_turn_corrected = False
