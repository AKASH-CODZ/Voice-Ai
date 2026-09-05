"""The sentence chunker decides when audio starts playing, so its edge cases
are latency-critical, not cosmetic."""

from __future__ import annotations

import pytest

from app.pipeline.chunker import SentenceChunker


def collect(tokens: list[str], **kwargs) -> list[str]:
    chunker = SentenceChunker(**kwargs)
    out: list[str] = []
    for token in tokens:
        out.extend(chunker.push(token))
    if (tail := chunker.flush()):
        out.append(tail)
    return out


def test_splits_on_sentence_boundaries():
    result = collect(["Hello there, good to meet you. What is your current role?"])
    assert result == ["Hello there, good to meet you.", "What is your current role?"]


def test_does_not_split_on_abbreviation():
    result = collect(["Dr. Smith runs the lab and knows the answer here. Next question."])
    assert result[0].startswith("Dr. Smith")
    assert "Smith" in result[0]


def test_does_not_split_on_decimal():
    result = collect(["We measured 3.5 percent improvement in the benchmark today."])
    assert len(result) == 1


def test_does_not_split_on_ellipsis():
    result = collect(["Wait... I think you mean something else entirely here."])
    assert len(result) == 1


def test_first_short_fragment_is_spoken_immediately():
    """The opening fragment gets its own TTS call on purpose.

    Waiting to accumulate 24 characters before the first synthesis measured
    377 ms → 1384 ms of extra time-to-first-audio when the model opened with a
    long sentence. Speaking "Sure." at once covers that gap.
    """
    result = collect(["Sure. ", "Let me explain the architecture in some detail."])
    assert result[0] == "Sure."
    assert len(result) == 2


def test_later_short_fragments_still_merge_forward():
    """After the first chunk the latency pressure is off, so a two-word
    fragment is no longer worth its own synthesis round-trip."""
    result = collect([
        "This opening sentence is comfortably long enough. ",
        "Yes. ",
        "Here is the substantive follow up to that point.",
    ])
    assert result[0] == "This opening sentence is comfortably long enough."
    assert result[1].startswith("Yes.")
    assert "substantive" in result[1]


def test_runon_breaks_at_word_boundary():
    text = "this is a very long run on sentence with absolutely no punctuation anywhere in it at all"
    result = collect([text], max_chars=60)
    assert len(result) > 1
    # No chunk may end mid-word: rejoining must reproduce the original words.
    assert " ".join(result).split() == text.split()


def test_streaming_preserves_every_word():
    """Chunk boundaries legitimately differ by arrival pattern — a sentence
    that closes at the end of the buffer is emitted immediately rather than
    held back, because time-to-first-audio is what the user feels. What must
    NOT differ is the content: no word may be dropped or duplicated."""
    text = "First sentence here. Second one follows. Third and last one."
    streamed = collect(list(text))
    at_once = collect([text])
    assert " ".join(streamed).split() == text.split()
    assert " ".join(at_once).split() == text.split()


def test_streaming_emits_first_sentence_early():
    """Regression guard on the latency behaviour above."""
    chunker = SentenceChunker()
    out: list[str] = []
    for ch in "First sentence here.":
        out.extend(chunker.push(ch))
    assert out == ["First sentence here."], "first sentence must not wait for the second"


@pytest.mark.parametrize("terminator", [".", "!", "?"])
def test_all_terminators_split(terminator: str):
    result = collect([f"This is long enough to be a chunk{terminator} And this is the next one."])
    assert len(result) == 2


def test_flush_drains_unterminated_tail():
    chunker = SentenceChunker()
    assert chunker.push("An unfinished thought with no period") == []
    assert chunker.flush() == "An unfinished thought with no period"
    assert chunker.flush() is None
