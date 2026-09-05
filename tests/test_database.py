"""SQLite layer, IP-bound sessions, and the markdown export contract."""

from __future__ import annotations

import pytest

from app.core import export
from app.core.database import Database
from app.core.schemas import Mode, Role


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "t.db"))
    await database.connect()
    yield database
    await database.close()


async def test_wal_mode_is_enabled(db):
    """WAL is what lets the transcript API and MCP server read while the
    telemetry task writes."""
    cur = await db.conn.execute("PRAGMA journal_mode;")
    assert (await cur.fetchone())[0].lower() == "wal"


async def test_same_ip_maps_to_stable_user_id(db):
    first = await db.upsert_user("192.168.1.50")
    second = await db.upsert_user("192.168.1.50")
    other = await db.upsert_user("192.168.1.51")
    assert first == second
    assert first != other


async def test_session_lifecycle(db):
    user = await db.upsert_user("10.0.0.1")
    sid = await db.create_session(user, Mode.TEACHING, "local", topic="interview")

    session = await db.get_session(sid)
    assert session.mode is Mode.TEACHING
    assert session.topic == "interview"
    assert session.ended_at is None

    await db.end_session(sid)
    assert (await db.get_session(sid)).ended_at is not None


async def test_resumable_session_lookup(db):
    user = await db.upsert_user("10.0.0.2")
    sid = await db.create_session(user, Mode.CASUAL, "cloud")

    assert (await db.latest_active_session(user)).session_id == sid
    await db.end_session(sid)
    assert await db.latest_active_session(user) is None


async def test_turns_round_trip_with_flags(db):
    user = await db.upsert_user("10.0.0.3")
    sid = await db.create_session(user, Mode.TEACHING, "local")

    await db.add_turn(sid, Role.USER, "I am working here since two years",
                      hesitation=True, stt_ms=140.0)
    await db.add_turn(sid, Role.ASSISTANT, "Small fix: we'd say I have been working.",
                      correction_made=True, llm_ms=180.0, tts_ms=210.0, e2e_ms=530.0)

    turns = await db.get_turns(sid)
    assert [t.role for t in turns] == [Role.USER, Role.ASSISTANT]
    assert turns[0].hesitation is True
    assert turns[1].correction_made is True
    assert turns[1].e2e_ms == 530.0


async def test_recent_turns_are_oldest_first(db):
    """This ordering feeds the LLM context window — reversed history would make
    the model answer the wrong question."""
    user = await db.upsert_user("10.0.0.4")
    sid = await db.create_session(user, Mode.CASUAL, "cloud")
    for i in range(10):
        await db.add_turn(sid, Role.USER, f"message {i}")

    recent = await db.recent_turns(sid, 3)
    assert [t.text for t in recent] == ["message 7", "message 8", "message 9"]


async def test_deleting_session_cascades_to_turns(db):
    user = await db.upsert_user("10.0.0.5")
    sid = await db.create_session(user, Mode.CASUAL, "cloud")
    await db.add_turn(sid, Role.USER, "hello")

    assert await db.delete_session(sid) is True
    assert await db.get_turns(sid) == []


async def test_deleting_user_cascades_to_sessions(db):
    user = await db.upsert_user("10.0.0.6")
    sid = await db.create_session(user, Mode.CASUAL, "cloud")
    await db.delete_user_data(user)
    assert await db.get_session(sid) is None


async def test_user_eviction_honours_cap(db, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_tracked_users", 3)
    for i in range(6):
        await db.upsert_user(f"10.1.1.{i}")

    cur = await db.conn.execute("SELECT COUNT(*) FROM users")
    assert (await cur.fetchone())[0] <= 3


async def test_session_stats_aggregate(db):
    user = await db.upsert_user("10.0.0.7")
    sid = await db.create_session(user, Mode.TEACHING, "local")
    await db.add_turn(sid, Role.USER, "a", hesitation=True, e2e_ms=500.0)
    await db.add_turn(sid, Role.ASSISTANT, "b", correction_made=True, e2e_ms=700.0)

    stats = await db.session_stats(sid)
    assert stats["turns"] == 2
    assert stats["hesitations"] == 1
    assert stats["corrections"] == 1
    assert stats["avg_e2e_ms"] == 600.0


# ── export contract ──────────────────────────────────────────

async def test_export_is_strictly_user_and_ai(db):
    """The spec is explicit: 'just a .md of user: , Ai: and nothing more'."""
    user = await db.upsert_user("10.0.0.8")
    sid = await db.create_session(user, Mode.TEACHING, "local", topic="interview prep")
    await db.add_turn(sid, Role.USER, "Hello, I want to practice.", hesitation=True, stt_ms=120.0)
    await db.add_turn(sid, Role.ASSISTANT, "Great. What is your current role?",
                      correction_made=True, e2e_ms=540.0)

    md = await export.render_markdown(db, sid)

    assert "User: Hello, I want to practice." in md
    assert "AI: Great. What is your current role?" in md

    # No analysis may leak into the transcript, even though every one of these
    # values is sitting in the same rows we just read.
    lowered = md.lower()
    for leak in ("hesitat", "correction", "latency", "e2e", "stt", "logprob", "120", "540"):
        assert leak not in lowered, f"analysis leaked into transcript: {leak!r}"


async def test_export_collapses_multiline_turns(db):
    user = await db.upsert_user("10.0.0.9")
    sid = await db.create_session(user, Mode.CASUAL, "cloud")
    await db.add_turn(sid, Role.USER, "line one\nline two\n\nline three")

    md = await export.render_markdown(db, sid)
    assert "User: line one line two line three" in md


async def test_export_handles_empty_session(db):
    user = await db.upsert_user("10.0.0.10")
    sid = await db.create_session(user, Mode.CASUAL, "cloud")
    md = await export.render_markdown(db, sid)
    assert "No speech was recorded" in md


async def test_export_missing_session_raises(db):
    with pytest.raises(KeyError):
        await export.render_markdown(db, "s_does_not_exist")
