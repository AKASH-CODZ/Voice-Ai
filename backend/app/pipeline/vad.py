"""Silero VAD, the utterance segmenter, and the hesitation watchdog.

Why this file is more than a thin ONNX wrapper
----------------------------------------------
Raw VAD output is a per-frame speech probability. Turning that into "the user
finished a sentence" or "the user is stuck" needs three things the model does
not give you, and getting any of them wrong is what makes a voice agent feel
broken:

1. **Pre-roll.** By the time the VAD is confident that speech started, the first
   80–150 ms is already gone — and that is exactly where the plosive that
   distinguishes "pat" from "bat" lives. We keep a ring buffer of the frames
   *before* the trigger and prepend them to the utterance.

2. **Hysteresis.** A single threshold flaps on every breath. We use separate
   enter/exit thresholds and require a sustained run of silence
   (``END_OF_UTTERANCE_SILENCE_MS``, default 800 ms) before closing an
   utterance. This is the "VAD clipping" correction from the plan.

3. **A separate, much longer stall timer.** The plan flags "panic
   misdiagnosis" — a user drawing breath must not read as panic. The stall
   watchdog is a *different* timer at a *different* timescale (4 s), it only
   arms after the assistant has finished speaking, and it fires at most once
   per turn.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

from app.core.config import settings

log = logging.getLogger(__name__)

# Silero v5 consumes exactly 512 new samples at 16 kHz (32 ms) per call...
SILERO_WINDOW = 512
# ...but the ONNX graph expects those 512 samples PREFIXED with the last 64
# samples of the previous window, for 576 total. This is not optional and it is
# not documented in the model signature — the input is declared `[None, None]`,
# so passing a bare 512 is accepted silently and returns near-zero for
# everything, including obvious speech. Getting this wrong makes the agent
# simply never hear you. Verified empirically: real speech scores 0.93 mean
# with the context, 0.003 without it.
SILERO_CONTEXT = 64


class SileroVAD:
    """ONNX Silero VAD. Handles both the v4 (h/c) and v5 (state) signatures."""

    def __init__(self, model_path: Path | None = None, sample_rate: int | None = None) -> None:
        import onnxruntime as ort

        self.sample_rate = sample_rate or settings.sample_rate
        path = Path(model_path or settings.silero_vad_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {path}. Run `make models` (or "
                f"`python tools/download_models.py`) to fetch it."
            )

        opts = ort.SessionOptions()
        # VAD is tiny and runs every 32 ms; extra threads cost more in context
        # switching than they save in compute, and they steal cores from STT.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._v5 = "state" in self._input_names
        self.reset()
        log.info("Silero VAD loaded (%s signature) from %s", "v5" if self._v5 else "v4", path)

    def reset(self) -> None:
        self._context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame in [-1, 1]."""
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.shape[-1] != SILERO_WINDOW:
            frame = _fit(frame, SILERO_WINDOW)

        x = frame.reshape(1, -1)
        sr = np.array(self.sample_rate, dtype=np.int64)

        if self._v5:
            # Prepend the tail of the previous window (see SILERO_CONTEXT).
            x = np.concatenate([self._context, x], axis=1)
            out, self._state = self.session.run(
                None, {"input": x, "state": self._state, "sr": sr}
            )
            self._context = x[:, -SILERO_CONTEXT:]
        else:
            out, self._h, self._c = self.session.run(
                None, {"input": x, "h": self._h, "c": self._c, "sr": sr}
            )
        return float(np.asarray(out).ravel()[0])


def _fit(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[-1] > size:
        return frame[..., :size]
    return np.pad(frame, (0, size - frame.shape[-1]))


class UtteranceState(Enum):
    IDLE = auto()       # nothing happening
    SPEAKING = auto()   # user is mid-utterance
    TRAILING = auto()   # user stopped, we're waiting out the silence window


@dataclass(slots=True)
class VadDecision:
    """What the segmenter concluded about the frame it just consumed."""

    speech_prob: float
    is_speech: bool
    speech_started: bool = False
    utterance: np.ndarray | None = None   # set exactly once, on end-of-utterance
    stalled_ms: int | None = None         # set once when the stall watchdog fires


class UtteranceSegmenter:
    """Frames in, complete utterances out — plus a stall signal.

    Feed it 32 ms float32 frames via :meth:`push`. It returns a
    :class:`VadDecision` each time; when an utterance completes, the decision
    carries the full audio including pre-roll, ready for STT.
    """

    def __init__(
        self,
        vad: SileroVAD | None = None,
        *,
        enter_threshold: float = 0.55,
        exit_threshold: float = 0.35,
        min_utterance_ms: int = 250,
    ) -> None:
        self.vad = vad or SileroVAD()
        self.enter_threshold = enter_threshold
        # Deliberately lower than enter_threshold: once we believe someone is
        # talking we stay committed through brief dips rather than chopping
        # mid-word. Classic Schmitt-trigger hysteresis.
        self.exit_threshold = exit_threshold

        self.frame_ms = settings.frame_ms
        self._min_frames = max(1, min_utterance_ms // self.frame_ms)
        self._silence_frames_to_end = max(1, settings.end_of_utterance_silence_ms // self.frame_ms)
        self._stall_frames = max(1, settings.stall_silence_ms // self.frame_ms)

        preroll_frames = max(1, settings.preroll_ms // self.frame_ms)
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_frames)

        self.state = UtteranceState.IDLE
        self._buffer: list[np.ndarray] = []
        self._silence_run = 0
        self._speech_frames = 0

        # Stall watchdog — armed by the orchestrator when the AI stops talking.
        self._stall_armed = False
        self._stall_fired = False
        self._idle_run = 0

    # ── stall watchdog control ───────────────────────────────

    def arm_stall_watchdog(self) -> None:
        """Called when the assistant finishes speaking. Only from that moment
        does user silence mean 'stuck' rather than 'listening to the AI'."""
        self._stall_armed = True
        self._stall_fired = False
        self._idle_run = 0

    def disarm_stall_watchdog(self) -> None:
        self._stall_armed = False
        self._idle_run = 0

    # ── main loop ────────────────────────────────────────────

    def push(self, frame: np.ndarray) -> VadDecision:
        prob = self.vad(frame)
        threshold = self.exit_threshold if self.state is UtteranceState.SPEAKING else self.enter_threshold
        is_speech = prob >= threshold

        decision = VadDecision(speech_prob=prob, is_speech=is_speech)

        if is_speech:
            self._idle_run = 0
            if self.state is UtteranceState.IDLE:
                # Prepend the ring buffer so the utterance keeps its onset.
                self._buffer = list(self._preroll)
                self._preroll.clear()
                self.state = UtteranceState.SPEAKING
                self._speech_frames = 0
                decision.speech_started = True
                self.disarm_stall_watchdog()
            elif self.state is UtteranceState.TRAILING:
                # False alarm — they were only pausing between words.
                self.state = UtteranceState.SPEAKING

            self._buffer.append(frame)
            self._speech_frames += 1
            self._silence_run = 0
            return decision

        # ── silence ──
        if self.state is UtteranceState.SPEAKING:
            self.state = UtteranceState.TRAILING
            self._silence_run = 1
            self._buffer.append(frame)   # keep trailing silence; Whisper likes it
            return decision

        if self.state is UtteranceState.TRAILING:
            self._silence_run += 1
            self._buffer.append(frame)
            if self._silence_run >= self._silence_frames_to_end:
                decision.utterance = self._close_utterance()
            return decision

        # IDLE + silence: feed the pre-roll ring and run the stall watchdog.
        self._preroll.append(frame)
        if self._stall_armed and not self._stall_fired:
            self._idle_run += 1
            if self._idle_run >= self._stall_frames:
                self._stall_fired = True
                decision.stalled_ms = self._idle_run * self.frame_ms
        return decision

    def _close_utterance(self) -> np.ndarray | None:
        audio = np.concatenate(self._buffer) if self._buffer else None
        speech_frames = self._speech_frames

        self._buffer = []
        self._silence_run = 0
        self._speech_frames = 0
        self.state = UtteranceState.IDLE
        self.vad.reset()

        # Drop blips: a door slam or a cough clears the threshold for two
        # frames and would otherwise cost a full STT round-trip on noise.
        if audio is None or speech_frames < self._min_frames:
            return None
        return audio

    def reset(self) -> None:
        self._buffer = []
        self._preroll.clear()
        self._silence_run = 0
        self._speech_frames = 0
        self._idle_run = 0
        self.state = UtteranceState.IDLE
        self._stall_fired = False
        self.vad.reset()
