"""Cloud pipeline: Groq STT + Groq LLM + CPU TTS.

This is the "drop-and-go" path from the spec: a reviewer with no GPU sets one
environment variable and gets the same conversation, same modes, same logging.

TTS choice (see docs/DECISIONS.md → D-07)
-----------------------------------------
Groq has no general TTS tier we can rely on, so the cloud profile keeps TTS
*local to the container* and CPU-only:

* ``kokoro-onnx`` (default) — the same 82M model as the GPU path, run on CPU.
  Renders faster than real-time on any modern core, emits float32 PCM directly,
  needs no network round-trip, and gives the cloud demo an identical voice to
  the local build. On a free Hugging Face CPU Space this is the only option
  that works without adding a second vendor.
* ``edge`` — Microsoft Edge TTS. Free and higher fidelity, but it returns
  **MP3**, which we must decode to PCM before it can go on the binary channel.
  That decode needs ffmpeg on PATH; the cloud Docker image installs it, and a
  bare-metal run without it fails loudly at load time rather than silently
  producing noise.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import AsyncIterator

import numpy as np

from app.core.audio import pcm16_to_float32, resample, to_wav_bytes
from app.core.config import settings
from app.engines.base import LLMEngine, STTEngine, TTSEngine, Transcript, VoiceEngine

log = logging.getLogger(__name__)

GROQ_STT_RATE = 16_000


def _require_key() -> str:
    key = settings.groq_api_key.strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. The cloud engine cannot start without it — "
            "either add the key to .env or run with a GPU and ENGINE_MODE=local."
        )
    return key


class GroqSTT(STTEngine):
    name = "groq-whisper"

    def __init__(self) -> None:
        self._client = None

    async def load(self) -> None:
        if self._client is not None:
            return
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=_require_key(), max_retries=1)
        log.info("Groq STT ready — %s", settings.groq_stt_model)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        if self._client is None:
            await self.load()

        audio = resample(audio, sample_rate, GROQ_STT_RATE)
        wav = to_wav_bytes(audio, GROQ_STT_RATE)
        started = time.perf_counter()

        result = await self._client.audio.transcriptions.create(
            file=("utterance.wav", wav, "audio/wav"),
            model=settings.groq_stt_model,
            language="en",
            response_format="verbose_json",
            temperature=0.0,
        )

        segments = getattr(result, "segments", None) or []
        avg_lp = None
        no_speech = None
        if segments:
            def _mean(key: str) -> float | None:
                vals = [
                    s.get(key) if isinstance(s, dict) else getattr(s, key, None)
                    for s in segments
                ]
                vals = [v for v in vals if v is not None]
                return sum(vals) / len(vals) if vals else None

            avg_lp = _mean("avg_logprob")
            no_speech = _mean("no_speech_prob")

        return Transcript(
            text=(getattr(result, "text", "") or "").strip(),
            language="en",
            duration_ms=1000.0 * len(audio) / GROQ_STT_RATE,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            avg_logprob=avg_lp,
            no_speech_prob=no_speech,
        )

    async def unload(self) -> None:
        self._client = None


class GroqLLM(LLMEngine):
    name = "groq-llama"

    def __init__(self) -> None:
        self._client = None

    async def load(self) -> None:
        if self._client is not None:
            return
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=_require_key(), max_retries=1)
        log.info("Groq LLM ready — %s", settings.groq_llm_model)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if self._client is None:
            await self.load()

        stream = await self._client.chat.completions.create(
            model=settings.groq_llm_model,
            messages=messages,
            stream=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=220,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def unload(self) -> None:
        self._client = None


class KokoroCPUTTS(TTSEngine):
    """Kokoro-82M forced onto the CPU ONNX provider."""

    name = "kokoro-cpu"
    sample_rate = 24_000

    def __init__(self) -> None:
        self._kokoro = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if self._kokoro is not None:
            return
        from kokoro_onnx import Kokoro

        model_path, voices_path = settings.kokoro_model_path, settings.kokoro_voices_path
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro weights missing ({model_path}, {voices_path}). Run `make models`, "
                "or set CLOUD_TTS_PROVIDER=edge to use Microsoft Edge TTS instead."
            )

        def _load():
            return Kokoro(str(model_path), str(voices_path))

        self._kokoro = await asyncio.to_thread(_load)
        await self._warm()
        log.info("Kokoro (CPU) ready — voice %s", settings.kokoro_voice)

    async def _warm(self) -> None:
        """Burn the first inference at startup, not on the user's first turn.

        Measured on CPU: the first `create_stream` call costs ~1430 ms (RTF
        1.18) while every call after it runs at ~500 ms (RTF ~0.3). That gap is
        ONNX graph warm-up — kernel selection, arena allocation, weight
        paging — and it lands squarely on time-to-first-audio for the opening
        greeting unless we pay it here.
        """
        try:
            async for _ in self.synthesize("Ready."):
                pass
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS warm-up failed (%s) — first turn will be slower", exc)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Streamed synthesis — see the note on the local KokoroTTS."""
        if self._kokoro is None:
            await self.load()

        async with self._lock:
            async for samples, rate in self._kokoro.create_stream(
                text, voice=settings.kokoro_voice, speed=1.0, lang="en-us"
            ):
                audio = np.asarray(samples, dtype=np.float32)
                if rate != self.sample_rate:
                    audio = resample(audio, rate, self.sample_rate)
                slice_len = int(self.sample_rate * 0.12)
                for start in range(0, len(audio), slice_len):
                    yield audio[start:start + slice_len]

    async def unload(self) -> None:
        self._kokoro = None


class EdgeTTS(TTSEngine):
    """Microsoft Edge TTS, decoded MP3 → PCM through ffmpeg.

    We stream the MP3 into ffmpeg's stdin and read PCM out of stdout, so audio
    starts flowing before the full sentence has been synthesised. Buffering the
    whole MP3 first would hand back the latency the chunker just bought us.
    """

    name = "edge-tts"
    sample_rate = 24_000

    async def load(self) -> None:
        import edge_tts  # noqa: F401 — presence check only

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "CLOUD_TTS_PROVIDER=edge needs ffmpeg on PATH to decode MP3 to PCM. "
                "Install ffmpeg, or set CLOUD_TTS_PROVIDER=kokoro-onnx (no external "
                "binary required)."
            )
        log.info("Edge TTS ready — voice %s", settings.edge_tts_voice)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        import edge_tts

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate), "-ac", "1",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.stdin is not None and proc.stdout is not None

        async def _feed() -> None:
            try:
                communicate = edge_tts.Communicate(text, settings.edge_tts_voice)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        proc.stdin.write(chunk["data"])
                        await proc.stdin.drain()
            except Exception as exc:  # noqa: BLE001
                log.error("Edge TTS stream failed: %s", exc)
            finally:
                try:
                    proc.stdin.close()
                    await proc.stdin.wait_closed()
                except Exception:  # noqa: BLE001
                    pass

        feeder = asyncio.create_task(_feed(), name="edge-tts-feed")
        chunk_bytes = int(self.sample_rate * 0.12) * 2
        try:
            while True:
                pcm = await proc.stdout.read(chunk_bytes)
                if not pcm:
                    break
                yield pcm16_to_float32(pcm)
        finally:
            # Barge-in cancels this generator; make sure ffmpeg dies with it
            # instead of leaking a process per interrupted sentence.
            feeder.cancel()
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await asyncio.gather(feeder, proc.wait(), return_exceptions=True)


def build_cloud_engine() -> VoiceEngine:
    tts: TTSEngine = EdgeTTS() if settings.cloud_tts_provider == "edge" else KokoroCPUTTS()
    return VoiceEngine(
        kind="cloud",
        label="Cloud Accelerated (Groq)",
        stt=GroqSTT(),
        llm=GroqLLM(),
        tts=tts,
    )
