#!/usr/bin/env python3
"""Measure real per-stage latency on THIS machine.

The plan targets a sub-second loop. Whether you actually get one depends
entirely on your hardware, so this measures rather than asserts: it runs each
stage the same way the live pipeline does and reports what your box does.

    python tools/bench_latency.py                # whichever engine routing picks
    python tools/bench_latency.py --engine cloud # force Groq
    python tools/bench_latency.py --turns 5

The number that matters is **e2e**: the gap between the user finishing a
sentence and the first syllable coming back. Everything else is diagnosis.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.audio import duration_ms, resample  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.hardware import probe  # noqa: E402
from app.engines.base import VoiceEngine  # noqa: E402
from app.engines.cloud import build_cloud_engine  # noqa: E402
from app.engines.local import build_local_engine  # noqa: E402
from app.pipeline.chunker import SentenceChunker  # noqa: E402

PHRASES = [
    "I want to practice for a backend engineering interview today.",
    "I built a real time voice agent using Python and web sockets.",
    "The hardest part was keeping the latency under one second.",
    "Can you ask me something about database design?",
    "I think the bottleneck is in the text to speech stage.",
]

TARGET_E2E_MS = 800.0


def make_speech(text: str, path: Path, sample_rate: int) -> np.ndarray | None:
    """Real speech via macOS `say`. Returns None where that is unavailable —
    a synthetic tone would make the STT numbers meaningless."""
    if not path.exists():
        if sys.platform != "darwin" or shutil.which("say") is None:
            return None
        subprocess.run(
            ["say", "-o", str(path), f"--data-format=LEI16@{sample_rate}", text],
            check=True, capture_output=True,
        )
    with wave.open(str(path)) as fh:
        pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


async def bench_turn(engine: VoiceEngine, audio: np.ndarray) -> dict[str, float]:
    """One full turn, instrumented exactly like the orchestrator measures it."""
    started = time.perf_counter()

    transcript = await engine.stt.transcribe(audio, settings.sample_rate)
    stt_ms = transcript.latency_ms

    messages = [
        {"role": "system", "content": "You are a spoken voice agent. Reply in one or two short sentences. No markdown."},
        {"role": "user", "content": transcript.text or "Hello."},
    ]

    chunker = SentenceChunker()
    llm_started = time.perf_counter()
    ttft_ms = 0.0
    first_audio_ms = 0.0
    tts_ms = 0.0
    got_first_token = False
    spoke = False

    async for delta in engine.llm.stream(messages):
        if not got_first_token:
            ttft_ms = (time.perf_counter() - llm_started) * 1000
            got_first_token = True
        for chunk in chunker.push(delta):
            tts_started = time.perf_counter()
            async for _ in engine.tts.synthesize(chunk):
                if not spoke:
                    tts_ms = (time.perf_counter() - tts_started) * 1000
                    first_audio_ms = (time.perf_counter() - started) * 1000
                    spoke = True
                break  # first chunk is all we need to time
            break

    llm_ms = (time.perf_counter() - llm_started) * 1000
    if not spoke:  # very short reply, no sentence boundary reached
        tail = chunker.flush() or "Okay."
        tts_started = time.perf_counter()
        async for _ in engine.tts.synthesize(tail):
            tts_ms = (time.perf_counter() - tts_started) * 1000
            first_audio_ms = (time.perf_counter() - started) * 1000
            break

    return {
        "stt_ms": stt_ms, "ttft_ms": ttft_ms, "llm_ms": llm_ms,
        "tts_ms": tts_ms, "e2e_ms": first_audio_ms,
        "audio_ms": duration_ms(audio, settings.sample_rate),
        "text": transcript.text,
    }


def summarise(label: str, values: list[float]) -> str:
    if not values:
        return f"  {label:<8} —"
    return (
        f"  {label:<8} mean {statistics.fmean(values):7.0f} ms   "
        f"min {min(values):6.0f}   max {max(values):6.0f}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark EchoSync pipeline latency.")
    parser.add_argument("--engine", choices=["auto", "local", "cloud"], default="auto")
    parser.add_argument("--turns", type=int, default=len(PHRASES))
    args = parser.parse_args()

    report = probe()
    target = args.engine if args.engine != "auto" else report.recommended_engine

    print("\n\033[1mEchoSync latency benchmark\033[0m")
    print("─" * 66)
    print(f"  engine   {target}")
    print(f"  reason   {report.reason}")
    gpu = report.primary_gpu
    print(f"  hardware {gpu.name if gpu else ('Apple Silicon' if report.apple_silicon else report.cpu_model)}")
    if target == "local":
        print(f"  whisper  {settings.whisper_model} on {settings.resolved_whisper_device}")
        print(f"  llm      {settings.ollama_model}")

    engine = build_local_engine(gpu.name if gpu else None) if target == "local" else build_cloud_engine()
    print("\n  loading engines…")
    load_started = time.perf_counter()
    await engine.load()
    print(f"  ready in {time.perf_counter() - load_started:.1f}s\n")

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"  {'#':<3} {'stt':>8} {'ttft':>8} {'tts':>8} {'e2e':>9}   transcript")
    print("  " + "─" * 78)

    results: list[dict[str, float]] = []
    for i, phrase in enumerate(PHRASES[: args.turns]):
        audio = make_speech(phrase, models_dir / f"_bench_{i}.wav", settings.sample_rate)
        if audio is None:
            print("  ! Cannot synthesise reference speech on this platform "
                  "(needs macOS `say`). Record 3-5 short WAVs at 16 kHz into "
                  "models/_bench_N.wav and re-run.")
            return 1

        audio = resample(audio, 16000, settings.sample_rate)
        result = await bench_turn(engine, audio)
        results.append(result)
        flag = "\033[32m✓\033[0m" if result["e2e_ms"] <= TARGET_E2E_MS else "\033[33m·\033[0m"
        print(f"  {i + 1:<3} {result['stt_ms']:>7.0f}m {result['ttft_ms']:>7.0f}m "
              f"{result['tts_ms']:>7.0f}m {result['e2e_ms']:>8.0f}m {flag} "
              f"{str(result['text'])[:44]}")

    print("\n\033[1m  Summary\033[0m")
    for key, label in [("stt_ms", "STT"), ("ttft_ms", "LLM TTFT"),
                       ("tts_ms", "TTS"), ("e2e_ms", "E2E")]:
        print(summarise(label, [r[key] for r in results]))

    e2e = [r["e2e_ms"] for r in results]
    mean_e2e = statistics.fmean(e2e)
    verdict = "\033[32mwithin\033[0m" if mean_e2e <= TARGET_E2E_MS else "\033[33mabove\033[0m"
    print(f"\n  Mean end-to-end {mean_e2e:.0f} ms — {verdict} the {TARGET_E2E_MS:.0f} ms target.")

    if mean_e2e > TARGET_E2E_MS:
        worst = max(
            [("STT", statistics.fmean([r["stt_ms"] for r in results])),
             ("LLM", statistics.fmean([r["ttft_ms"] for r in results])),
             ("TTS", statistics.fmean([r["tts_ms"] for r in results]))],
            key=lambda kv: kv[1],
        )
        print(f"  Dominant stage: {worst[0]} at {worst[1]:.0f} ms.")
        hints = {
            "STT": "Try a smaller Whisper model (tiny.en) or move it to CUDA.",
            "LLM": "Ensure the model is warm (keep_alive) or route the LLM to Groq.",
            "TTS": "Move Kokoro to GPU (KOKORO_DEVICE=cuda + onnxruntime-gpu).",
        }
        print(f"  → {hints[worst[0]]}")

    await engine.unload()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
