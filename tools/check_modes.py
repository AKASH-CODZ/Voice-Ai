#!/usr/bin/env python3
"""Judge the three conversation modes against the real Ollama model.

Voice quality cannot be signed off from a coder model (D-13). This hits
llama3.2:3b-instruct-q4_K_M the same way the live pipeline does and checks:

* Casual — repairs via cross-question, never names the mistake
* Teaching — one explicit correction, and a stall produces a helpful hint
* Observation — no praise, no correction, no stall hint

    python tools/check_modes.py
    python tools/check_modes.py --model llama3.2:3b-instruct-q4_K_M
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.core.schemas import Mode  # noqa: E402
from app.pipeline.state import ConversationState, _CORRECTION_MARKERS  # noqa: E402

BROKEN = "I am working in this company since two years and I am knowing Python good."
PRAISE = re.compile(
    r"\b(great(?: answer| job)?|good (?:job|point|answer)|well done|excellent|"
    r"that's a strong|strong example|nice work)\b",
    re.IGNORECASE,
)
STALL_LEAK = re.compile(
    r"\b(you('ve| have) been silent|you('re| are) stuck|don't (panic|worry)|"
    r"take your time|user_stalled)\b",
    re.IGNORECASE,
)


async def chat(messages: list[dict[str, str]], model: str, base: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 120, "num_ctx": 4096},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        return ((r.json().get("message") or {}).get("content") or "").strip()


def _ok(label: str, passed: bool, detail: str) -> bool:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}: {detail}")
    return passed


async def run_mode(mode: Mode, model: str, base: str) -> bool:
    state = ConversationState(session_id=f"quality-{mode.value}", mode=mode, topic="backend interview")
    print(f"\n== {mode.value.upper()} ==")
    reply = await chat(state.build_messages(BROKEN), model, base)
    print(f"  reply: {reply!r}")
    state.append_user(BROKEN)
    state.append_assistant(reply)
    ok = True

    if mode is Mode.CASUAL:
        ok &= _ok("no explicit correction", not _CORRECTION_MARKERS.search(reply), reply[:120])
        ok &= _ok("asks something back", "?" in reply, "follow-up question expected")
    elif mode is Mode.TEACHING:
        ok &= _ok("makes a correction", bool(_CORRECTION_MARKERS.search(reply)), "expected 'we'd say' / 'small fix'")
        stall_msgs = state.build_messages(None)
        # Force the stall onto the wire the way the orchestrator does.
        state.note_stall(4000)
        stall_msgs = state.build_messages("um")
        stall_reply = await chat(stall_msgs, model, base)
        print(f"  stall: {stall_reply!r}")
        ok &= _ok(
            "stall hint is helpful not embarrassing",
            (not STALL_LEAK.search(stall_reply)) or "take your time" in stall_reply.lower(),
            stall_reply[:120],
        )
        ok &= _ok("stall does not pile on a correction lecture", stall_reply.count(".") <= 4, stall_reply[:120])
    elif mode is Mode.OBSERVATION:
        ok &= _ok("no correction", not _CORRECTION_MARKERS.search(reply), reply[:120])
        ok &= _ok("no praise", not PRAISE.search(reply), reply[:120])
        leaked = state.note_stall(4000)
        ok &= _ok("stall swallowed (never reaches LLM)", leaked is False, f"note_stall returned {leaked}")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--base", default=settings.ollama_base_url)
    args = parser.parse_args()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            tags = (await client.get(f"{args.base}/api/tags")).json()
    except Exception as exc:  # noqa: BLE001
        print(f"Ollama unreachable at {args.base}: {exc}")
        return 2

    names = [m.get("name") for m in tags.get("models", [])]
    if args.model not in names and not any(args.model in (n or "") for n in names):
        print(f"model {args.model} is not pulled. Have: {names}")
        print("Run: ollama pull llama3.2:3b-instruct-q4_K_M")
        return 2

    print(f"model={args.model}  prompt_user={BROKEN!r}")
    results = []
    for mode in (Mode.CASUAL, Mode.TEACHING, Mode.OBSERVATION):
        results.append(await run_mode(mode, args.model, args.base))
    print("\n" + ("ALL MODES PASSED" if all(results) else "SOME CHECKS FAILED — read the replies above"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
