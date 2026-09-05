"""Sentence-boundary chunker for streaming LLM output into TTS.

This is the single highest-leverage latency fix in the whole system, and it is
the "Waiting for the Full LLM Sentence" correction from the plan.

Naive pipeline:   [LLM generates 60 tokens] → [TTS renders] → speech
                  time-to-first-audio ≈ full generation + full synthesis

Chunked pipeline: [LLM emits "Hi there."] → TTS starts *now*, while the LLM
                  keeps generating sentence two in the background.
                  time-to-first-audio ≈ first sentence only

On a 3B local model that is roughly 900 ms down to 250 ms — the difference
between "laggy" and "real".

The subtlety is that a period is not reliably a sentence end. "Dr. Smith",
"3.5", "e.g." and "..." all contain terminators that must not trigger a flush,
because flushing mid-phrase gives the TTS no prosodic context and you hear the
seam.
"""

from __future__ import annotations

import re

# Abbreviations whose trailing period never ends a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "approx", "dept", "est", "fig", "inc", "ltd", "no", "vol", "al", "u.s",
    "a.m", "p.m",
}

_TERMINATORS = ".!?"
# A terminator ends a sentence when followed by whitespace/end AND not part of
# a decimal, an ellipsis, or a known abbreviation.
_SENTENCE_END = re.compile(
    r"""
    (?<![0-9])            # not a decimal point: "3.5"
    ([.!?]+)              # one or more terminators
    (["')\]]*)            # optional closing quote/bracket
    (?=\s|$)              # followed by whitespace or end of buffer
    """,
    re.VERBOSE,
)


class SentenceChunker:
    """Accumulates LLM deltas and yields speakable chunks.

    ``min_chars`` stops us shipping a two-word fragment ("Sure.") as its own
    TTS call — the per-call overhead would dominate. ``max_chars`` force-flushes
    a run-on sentence at a comma so a model that forgets punctuation can never
    stall the audio indefinitely.
    """

    def __init__(
        self,
        min_chars: int = 24,
        max_chars: int = 220,
        first_chunk_min_chars: int = 2,
        first_chunk_max_chars: int = 80,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        # The FIRST chunk of a reply gets a far lower bar. Measured: when the
        # model opens with a long sentence, first-audio cost jumped from 377 ms
        # to 1384 ms purely because we were waiting to accumulate more text.
        # Speaking "Great!" immediately buys ~400 ms of cover while the rest
        # renders underneath — which is exactly how the gap stops being
        # audible. Per-call synthesis overhead is a price worth paying once.
        self.first_chunk_min_chars = first_chunk_min_chars
        # ...and a much tighter ceiling. Lowering the floor alone did nothing
        # when the model opens with one long unpunctuated sentence: measured
        # 2004 ms to first audio because we sat waiting for a terminator that
        # was 200 characters away. Force-flushing the opening clause at a comma
        # or word boundary is what actually closes that gap. Later chunks keep
        # the generous 220-char ceiling, where prosody matters more than speed
        # because audio is already playing.
        self.first_chunk_max_chars = first_chunk_max_chars
        self._buf = ""
        self._emitted = 0

    def push(self, delta: str) -> list[str]:
        """Add a token delta; return any chunks that are ready to speak."""
        self._buf += delta
        chunks: list[str] = []

        while True:
            chunk = self._try_split()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    @property
    def _current_min(self) -> int:
        return self.first_chunk_min_chars if self._emitted == 0 else self.min_chars

    @property
    def _current_max(self) -> int:
        return self.first_chunk_max_chars if self._emitted == 0 else self.max_chars

    def _try_split(self) -> str | None:
        for match in _SENTENCE_END.finditer(self._buf):
            end = match.end()
            candidate = self._buf[:end].strip()

            # Too short to be worth its own synthesis call — but only if we
            # can see that more text follows. When the terminator sits at the
            # end of the buffer we do not know whether the model is done, and
            # waiting would trade guaranteed latency for a possible saving.
            # Emitting early is the right call: this is usually the FIRST
            # chunk, and time-to-first-audio is the number the user feels.
            if len(candidate) < self._current_min and end < len(self._buf):
                continue
            if _ends_with_abbreviation(candidate):
                continue

            self._buf = self._buf[end:].lstrip()
            self._emitted += 1
            return candidate

        # Run-on protection. Prefer a comma, then any word boundary; only cut
        # mid-word as a last resort, because a chunk that ends inside a word
        # makes the TTS mispronounce both halves.
        limit = self._current_max
        if len(self._buf) >= limit:
            floor = min(self._current_min, limit // 2)
            cut = self._buf.rfind(",", floor, limit)
            if cut > 0:
                cut += 1
            else:
                cut = self._buf.rfind(" ", floor, limit)
                if cut <= 0:
                    cut = limit
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if chunk:
                self._emitted += 1
            return chunk or None

        return None

    def flush(self) -> str | None:
        """Drain whatever is left when the LLM stream ends."""
        remainder = self._buf.strip()
        self._buf = ""
        if remainder:
            self._emitted += 1
        return remainder or None

    def reset(self) -> None:
        self._buf = ""
        self._emitted = 0

    @property
    def pending(self) -> str:
        return self._buf


def _ends_with_abbreviation(text: str) -> bool:
    if not text.endswith("."):
        return False
    # Ellipsis is a pause, not a sentence end.
    if text.endswith("..") :
        return True
    tail = text[:-1].rsplit(None, 1)
    if not tail:
        return False
    word = tail[-1].lower().strip("(\"'")
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())
