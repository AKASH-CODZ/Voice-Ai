"""Mode state machine — where the three modes actually differ."""

from __future__ import annotations

from app.core.schemas import Mode
from app.pipeline.state import ConversationState, load_prompt


def test_each_mode_loads_a_distinct_prompt():
    prompts = {mode: load_prompt(mode) for mode in Mode}
    assert len(set(prompts.values())) == 3
    for text in prompts.values():
        assert "EchoSync" in text  # shared block is always included


def test_observation_prompt_forbids_correction():
    text = load_prompt(Mode.OBSERVATION).lower()
    assert "never correct" in text


def test_teaching_prompt_handles_stall():
    assert "[USER_STALLED]" in load_prompt(Mode.TEACHING)


def test_only_teaching_intervenes_on_stall():
    for mode, expected in [
        (Mode.TEACHING, True),
        (Mode.CASUAL, False),
        (Mode.OBSERVATION, False),
    ]:
        state = ConversationState(session_id="s", mode=mode)
        assert state.note_stall(4200) is expected, f"{mode} intervention"
        # The flag is recorded for telemetry in every mode, regardless.
        assert state.last_turn_hesitated is True


def injected_stall_notices(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stall notices that were actually injected for this turn.

    The mode prompts themselves mention `[USER_STALLED]` (teaching explains how
    to respond to it; observation is told to ignore it), so a naive substring
    search over all messages always matches the system prompt at index 0.
    The injected notice is distinguishable by its colon and payload.
    """
    return [m for m in messages[1:] if "[USER_STALLED:" in m["content"]]


def test_stall_notice_is_a_system_message_not_user_text():
    """If the notice were appended to user text the model would read it aloud."""
    state = ConversationState(session_id="s", mode=Mode.TEACHING)
    state.note_stall(4500)
    messages = state.build_messages("um")

    stall = injected_stall_notices(messages)
    assert len(stall) == 1
    assert stall[0]["role"] == "system"
    assert "4.5 seconds" in stall[0]["content"]
    assert messages[-1] == {"role": "user", "content": "um"}


def test_stall_notice_is_consumed_once():
    state = ConversationState(session_id="s", mode=Mode.TEACHING)
    state.note_stall(4500)
    assert injected_stall_notices(state.build_messages("hi"))
    assert not injected_stall_notices(state.build_messages("hi again"))


def test_observation_never_emits_stall_notice():
    state = ConversationState(session_id="s", mode=Mode.OBSERVATION)
    state.note_stall(9000)
    assert not injected_stall_notices(state.build_messages("..."))


def test_greet_injects_a_user_kick_when_history_is_empty():
    """System-only chat returns empty from llama3.2-instruct; greet must not."""
    state = ConversationState(session_id="s", mode=Mode.CASUAL, topic="backend interview")
    messages = state.build_messages(None)
    assert messages[-1]["role"] == "user"
    assert "greet" in messages[-1]["content"].lower()
    assert state.history == []


def test_greet_kick_is_not_used_when_the_user_already_spoke():
    state = ConversationState(session_id="s", mode=Mode.CASUAL)
    messages = state.build_messages("hello there")
    assert messages[-1] == {"role": "user", "content": "hello there"}
    assert sum(1 for m in messages if m["role"] == "user") == 1


def test_stall_intervention_still_ends_on_a_user_turn():
    state = ConversationState(session_id="s", mode=Mode.TEACHING)
    state.append_user("I was talking about indexes.")
    state.append_assistant("Go on.")
    state.note_stall(4500)
    messages = state.build_messages(None)
    assert messages[-1]["role"] == "user"
    assert injected_stall_notices(messages)


def test_topic_is_injected_into_system_prompt():
    state = ConversationState(session_id="s", mode=Mode.CASUAL, topic="backend interview")
    assert "backend interview" in state.system_prompt()


def test_history_is_trimmed_to_context_window(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "context_window_turns", 3)
    state = ConversationState(session_id="s")
    for i in range(20):
        state.append_user(f"user {i}")
        state.append_assistant(f"assistant {i}")
    assert len(state.history) <= 6
    assert state.history[-1]["content"] == "assistant 19"


def test_correction_flag_only_set_in_teaching():
    teaching = ConversationState(session_id="s", mode=Mode.TEACHING)
    teaching.append_assistant("Small fix: we'd say 'I have been working'. Where do you work?")
    assert teaching.last_turn_corrected is True

    casual = ConversationState(session_id="s", mode=Mode.CASUAL)
    casual.append_assistant("Small fix: we'd say 'I have been working'. Where do you work?")
    assert casual.last_turn_corrected is False


def test_mode_switch_preserves_history():
    state = ConversationState(session_id="s", mode=Mode.CASUAL)
    state.append_user("hello")
    state.append_assistant("hi there")
    state.set_mode(Mode.OBSERVATION)
    assert state.mode is Mode.OBSERVATION
    assert len(state.history) == 2
    assert "never correct" in state.system_prompt().lower()
