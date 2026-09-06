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
Anyone who clones the repo. Sessions are IP-bound + `localStorage`, not
accounts — see `decisions.md` D-04. The author's two test laptops (Apple M4
and an RTX 5070) are independent hosts, not a split deployment.

## Non-goals
- Not multi-tenant SaaS. No auth, no billing, no admin panel.
- Not trying to match LiveKit/WebRTC-grade transport — see D-01.
- The `.md` transcript export is intentionally analysis-free (D-05); scoring
  and hesitation data live in SQLite only, queried separately.
- Not a multi-machine product. One clone runs on one computer.

## Current milestone
Voice pipeline is feature-complete and measured on two **independent** test
machines (M4 CPU local; RTX 5070 CUDA local at 530 ms E2E). Docker images
are written but not yet built on a daemon. Cloud (Groq) error paths are
tested; a live Groq key is still optional for contributors. See D-18 for
the clone → probe → route story.

## Original spec
`../../Hybrid AI Voice Agent Technical Plan.docx` (two levels up: this repo's
parent folder) is the source 3-round spec. Where the build deviates from it,
the deviation and why are in `decisions.md`, not here.
