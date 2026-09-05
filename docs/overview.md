# Overview

## What it is
EchoSync AI is a real-time spoken-English practice agent. You talk, it talks
back over a live audio stream, and it behaves differently depending on mode:

| Mode | Behavior |
|---|---|
| Casual | Natural chat; mistakes get repaired via cross-questions, never called out |
| Teaching | Active tutor; corrects one thing at a time, hints after 4s of silence |
| Observation | Mock interview; zero feedback, everything logged silently for later review |

The audio path, engines, and state machine are identical across modes — only
the system prompt and whether hesitation reaches the model differ. Adding a
fourth mode is a prompt file plus an enum member (`backend/app/pipeline/state.py`).

## Who uses it
Single user (Akash), for interview practice and spoken-English drilling.
Sessions are IP-bound, not account-based — see `decisions.md` D-04.

## Non-goals
- Not multi-tenant SaaS. No auth, no billing, no admin panel.
- Not trying to match LiveKit/WebRTC-grade transport — see D-01.
- The `.md` transcript export is intentionally analysis-free (D-05); scoring
  and hesitation data live in SQLite only, queried separately.

## Current milestone
Feature-complete and verified end-to-end on the dev machine (Apple M4,
CPU-only STT/TTS). **Not yet verified on the target GPU (RTX 5070), no Docker
image has been built, no real Groq API call has been made.** See `HANDOFF.md`
at the repo root for the live state and next steps.

## Original spec
`../../Hybrid AI Voice Agent Technical Plan.docx` (two levels up: this repo's
parent folder) is the source 3-round spec. Where the build deviates from it,
the deviation and why are in `decisions.md`, not here.
