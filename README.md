---
title: EchoSync AI
emoji: 🎙️
colorFrom: indigo
colorTo: rose
sdk: docker
app_port: 8000
pinned: false
short_description: Real-time spoken-English practice agent (cloud profile)
---

<p align="center">
  <img src="docs/assets/orb.gif" width="128" alt="A pulsing voice orb">
</p>

<h1 align="center">EchoSync AI</h1>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=26&duration=2800&pause=900&color=4CC9F0&center=true&vCenter=true&width=680&lines=Talk.+It+talks+back.;Practice+English+out+loud.;A+patient+partner+that+never+tires." alt="Talk. It talks back." />
</p>

<p align="center">
  You want to speak English more confidently.<br/>
  A practice partner is not always around.<br/>
  EchoSync sits with you, listens, and talks back.
</p>

<p align="center">
  <a href="https://echosync-web.onrender.com"><img src="https://img.shields.io/badge/Try_it_live-4cc9f0?style=for-the-badge" alt="Try it live"></a>
  &nbsp;
  <a href="#run-it-on-your-computer"><img src="https://img.shields.io/badge/Run_on_your_computer-8b5cf6?style=for-the-badge" alt="Run on your computer"></a>
</p>

<p align="center">
  <img src="docs/assets/wave.svg" width="680" alt="">
</p>

<p align="center">
  <img src="docs/assets/hero.jpg" width="880" alt="A glowing glass orb in a dark room">
</p>

---

## The room

<p align="center">
  <img src="docs/assets/ui.jpg" width="880" alt="EchoSync screen: modes, glowing orb, live transcript">
</p>

Pick a mood. Press start. Speak. It answers. A transcript writes itself on the side.

---

## Three moods

<p align="center">
  <img src="docs/assets/modes.jpg" width="880" alt="Casual, Teaching, and Observation">
</p>

| Casual | Teaching | Observation |
| --- | --- | --- |
| Chat like a friend. Mistakes are fixed with a question, never a scolding. | A gentle tutor. One fix at a time. If you freeze, it offers a hint. | A mock interview. No praise, no correction. Review the notes after. |

---

## How a turn feels

<p align="center">
  <img src="docs/assets/how-it-works.jpg" width="880" alt="You speak, it hears, it thinks, it talks back">
</p>

You stop talking → it replies.

About **half a second** on a gaming laptop. About **one second** on a Mac.

---

## Architecture

<p align="center">
  <img src="docs/assets/architecture.jpg" width="880" alt="Browser to EchoSync brain, then your computer or the cloud">
</p>

Same orb. Two engines. Your computer if it can. The cloud if it cannot.

```mermaid
flowchart LR
  You[You speak] --> App[EchoSync]
  App --> Local[Your laptop]
  App --> Cloud[Free demo]
  Local --> You2[You hear a reply]
  Cloud --> You2
```

---

## Tools

| Job | Tool |
| --- | --- |
| The screen | Next.js |
| The brain | FastAPI |
| Hears you | Whisper on your machine, or Groq on the demo |
| Thinks | Ollama (Llama 3.2) on your machine, or Groq on the demo |
| Speaks | Kokoro on your machine, or Edge TTS on the demo |
| Notices a pause | Silero |
| Remembers the chat | SQLite |
| Optional review in your editor | MCP |

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-black?logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=black" alt="Render">
</p>

---

## Where it lives

<p align="center">
  <img src="docs/assets/infra.jpg" width="880" alt="GitHub, Render, Hugging Face, your laptop, Groq">
</p>

| Place | What it is |
| --- | --- |
| [Try it live](https://echosync-web.onrender.com) | Free Render demo. Sleeps after 15 minutes of quiet. First open can take a minute. |
| [Hugging Face page](https://huggingface.co/spaces/Akash-8/EchoSync-AI) | Project page. No microphone on the free plan. |
| Your laptop | Private. No account. No key needed if Ollama is installed. |

One clone, one computer. There is nothing to wire together.

---

## Academic note

**Problem.** Spoken English needs a partner. Partners are scarce. Interviews are hard.

**Method.** A live voice loop: notice the end of a sentence, write it down, think of a reply, speak it. Three teaching styles. Runs on your machine when it can, and on a free cloud demo when it cannot.

**Result.** About **530 ms** to first sound on an RTX 5070 laptop. **93** automated tests. A public demo on Render.

---

## Try it live

**https://echosync-web.onrender.com**

Allow the microphone. Wait through the first wake-up if the demo was sleeping.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AKASH-CODZ/Voice-Ai)

Paste `GROQ_API_KEY` when asked. Do not commit it.

---

## Run it on your computer

```bash
git clone https://github.com/AKASH-CODZ/Voice-Ai.git
cd Voice-Ai
cp .env.example .env
make setup && make models
make backend          # :8000
make frontend         # :3000
```

Use **Python 3.11 or 3.12**.

If Ollama is installed, pull the chat model once:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```

No GPU? Add a Groq key to `.env` and the cloud path takes over.

<details>
<summary><strong>Docker, tests, and extra notes</strong></summary>

```bash
make up-cloud         # cloud path in Docker
make up-gpu           # CUDA host
make test             # pytest
make lint             # ruff + tsc
make bench            # five real turns on this machine
```

MCP (read-only session notes in your editor):

```bash
claude mcp add echosync -- python -m echosync_mcp.server
```

Design choices: [`docs/decisions.md`](docs/decisions.md).
</details>
