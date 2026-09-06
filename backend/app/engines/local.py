"""Local GPU pipeline: Faster-Whisper + Ollama + Kokoro.

Threading model
---------------
Faster-Whisper and Kokoro are synchronous, CPU/GPU-bound libraries. Calling
them directly on the event loop would freeze *every* other connection —
including the WebSocket that is trying to stream audio to the browser. Both are
therefore dispatched through ``asyncio.to_thread``. They release the GIL inside
their native inference calls, so this genuinely overlaps with I/O rather than
just moving the stall.

Ollama is HTTP, so it stays natively async via httpx.

VRAM budget on the 8 GB target (RTX 5070):
    Faster-Whisper base.en int8_float16   ~1.0 GB
    Llama-3.2-3B-Instruct Q4_K_M          ~2.5 GB   (in the Ollama process)
    Kokoro-82M                            ~0.5 GB   (CPU by default — see D-03)
    CUDA context + framework overhead     ~1.5 GB
                                          -------
                                          ~5.5 GB, leaving ~2.5 GB of headroom
"""

from __future__ import annotations

import asyncio
import ctypes
import glob
import json
import logging
import os
import site
import time
from collections.abc import AsyncIterator

import httpx
import numpy as np

from app.core.audio import resample
from app.core.config import settings
from app.engines.base import LLMEngine, STTEngine, TTSEngine, Transcript, VoiceEngine

log = logging.getLogger(__name__)

WHISPER_RATE = 16_000


class FasterWhisperSTT(STTEngine):
    name = "faster-whisper"

    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        def _load():
            return WhisperModel(
                settings.whisper_model,
                device=settings.resolved_whisper_device,
                compute_type=settings.resolved_whisper_compute_type,
            )

        log.info(
            "Loading Faster-Whisper %s on %s (%s)…",
            settings.whisper_model,
            settings.resolved_whisper_device,
            settings.resolved_whisper_compute_type,
        )
        self._model = await asyncio.to_thread(_load)
        await self._warm()
        log.info("Faster-Whisper ready")

    async def _warm(self) -> None:
        """Burn the first inference at startup, not on the user's first turn.

        Measured on an RTX 5070 (cuda / int8_float16): the first transcribe
        costs ~7165 ms while every call after it runs at ~100 ms — a 71x gap.
        That is CTranslate2 selecting CUDA kernels and bringing up its
        cuBLAS/cuDNN handles. On CPU the same gap is small enough to miss,
        which is why this was invisible until the code ran on the target GPU.
        Without this, the user's opening sentence takes ~8.7 s end-to-end.
        """
        try:
            # Faint noise rather than digital silence: the decoder short-circuits
            # on pure zeros, leaving its kernels cold.
            rng = np.random.default_rng(0)
            noise = (rng.standard_normal(WHISPER_RATE) * 1e-3).astype(np.float32)
            await self.transcribe(noise, WHISPER_RATE)
        except Exception as exc:  # noqa: BLE001
            log.warning("STT warm-up failed (%s) — first turn will be slower", exc)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        if self._model is None:
            await self.load()

        audio = resample(audio, sample_rate, WHISPER_RATE)
        started = time.perf_counter()

        def _run():
            segments, info = self._model.transcribe(
                audio,
                language="en",
                beam_size=1,              # greedy: beam search costs ~2x for a
                                          # negligible WER win on short turns
                vad_filter=False,         # our Silero segmenter already did this
                condition_on_previous_text=False,  # stops hallucination loops
                without_timestamps=True,
            )
            segments = list(segments)
            text = " ".join(s.text.strip() for s in segments).strip()
            avg_lp = (
                sum(s.avg_logprob for s in segments) / len(segments) if segments else None
            )
            no_speech = (
                sum(s.no_speech_prob for s in segments) / len(segments) if segments else None
            )
            return text, info, avg_lp, no_speech

        # One utterance at a time: concurrent transcribes on a single CTranslate2
        # model contend for the same GPU buffers and end up slower than serial.
        async with self._lock:
            text, info, avg_lp, no_speech = await asyncio.to_thread(_run)

        return Transcript(
            text=text,
            language=getattr(info, "language", "en"),
            duration_ms=1000.0 * len(audio) / WHISPER_RATE,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            avg_logprob=avg_lp,
            no_speech_prob=no_speech,
        )

    async def unload(self) -> None:
        self._model = None


class OllamaLLM(LLMEngine):
    name = "ollama"

    def __init__(self, model: str | None = None) -> None:
        self._client: httpx.AsyncClient | None = None
        self._model = model or settings.ollama_model

    async def load(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.ollama_base_url,
                timeout=httpx.Timeout(connect=2.0, read=60.0, write=10.0, pool=5.0),
            )

        # Warm the model. The first request after a cold start pays the full
        # weight-load cost (seconds on a 3B), and paying it here rather than on
        # the user's first sentence is the difference between a snappy demo and
        # an awkward silence. A missing/wrong tag must fail here so the router
        # can fall back to cloud *before* the user speaks (D-18).
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
                "keep_alive": "30m",
            },
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model {self._model} is not pulled. "
                f"Run: ollama pull {self._model}"
            )
        response.raise_for_status()
        log.info("Ollama warm: %s", self._model)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if self._client is None:
            await self.load()
        assert self._client is not None

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 220,       # spoken replies are short by design
                "repeat_penalty": 1.1,
                # Cap context explicitly; letting it grow silently is the
                # fastest way to blow the VRAM budget mid-conversation.
                "num_ctx": 4096,
            },
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = (event.get("message") or {}).get("content", "")
                if delta:
                    yield delta
                if event.get("done"):
                    break

    async def unload(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _enable_onnx_cuda() -> None:
    """Make Kokoro use the GPU. Two obstacles, both of which fail silently.

    1. kokoro-onnx 0.4.9 gates GPU on `find_spec("onnxruntime-gpu")`. Module
       names cannot contain hyphens — that is the *distribution* name, the
       module is `onnxruntime` — so it is always None and providers stay
       CPU-only no matter what is installed. Its `ONNX_PROVIDER` env var is
       the only supported way past this.
    2. The pip `nvidia-*` wheels put libcudnn/libcublas under
       site-packages/nvidia/*/lib, which the dynamic linker does not search,
       so the CUDA provider fails to load with "libcudnn.so.9: cannot open
       shared object file" and onnxruntime quietly falls back to CPU.
       Preloading them RTLD_GLOBAL resolves it in-process, which beats making
       every caller set LD_LIBRARY_PATH before starting Python.

    cuBLAS must load before cuDNN, which links against it.
    """
    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")

    # Preload the CUDA/cuDNN sonames ORT actually dlopens, in link order.
    # Globbing every lib*.so under nvidia/*/lib also pulls
    # libcudnn_engines_precompiled (~700 MB) — reckless inside WSL2's 7.6 GB
    # RAM cap. Do not add *engines* here. Measured miss if we stop at
    # libcudnn.so.9: "libcudnn_adv.so.9: cannot open shared object file"
    # and a silent CPU fallback.
    for soname in (
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_ext.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_tensor_ir.so.9",
        # ORT's CUDA provider DT_NEEDs this. mmap cost is real (~700 MB) but
        # without it the provider fails to load and silently falls back to CPU.
        "libcudnn_engines_precompiled.so.9",
    ):
        for base in site.getsitepackages():
            for lib in glob.glob(os.path.join(base, "nvidia", "*", "lib", soname)):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:  # noqa: PERF203 — best effort; verified after load
                    pass
                break


class KokoroTTS(TTSEngine):
    name = "kokoro"
    sample_rate = 24_000

    def __init__(self) -> None:
        self._kokoro = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if self._kokoro is not None:
            return
        from kokoro_onnx import Kokoro

        model_path = settings.kokoro_model_path
        voices_path = settings.kokoro_voices_path
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro weights missing ({model_path}, {voices_path}). "
                "Run `make models` to fetch them."
            )

        def _load():
            if settings.kokoro_device == "cuda":
                _enable_onnx_cuda()
            return Kokoro(str(model_path), str(voices_path))

        log.info("Loading Kokoro TTS (device=%s)…", settings.kokoro_device)
        self._kokoro = await asyncio.to_thread(_load)

        # Verify rather than assume. onnxruntime falls back to CPU silently
        # when a provider fails to initialise, and `get_available_providers()`
        # keeps listing CUDA regardless — it reports the build, not what loaded.
        # Only the session knows the truth. Measured cost of getting this wrong
        # on an RTX 5070: TTS 203 ms on GPU vs 901 ms on the CPU fallback.
        active = list(getattr(self._kokoro, "sess", None).get_providers()) if getattr(
            self._kokoro, "sess", None) is not None else []
        if settings.kokoro_device == "cuda" and "CUDAExecutionProvider" not in active:
            log.warning(
                "KOKORO_DEVICE=cuda but Kokoro is running on %s. Install "
                "onnxruntime-gpu and nvidia-cudnn-cu12 (see requirements-local.txt); "
                "TTS will be ~4x slower until then.",
                active or "CPU",
            )
        await self._warm()
        log.info("Kokoro ready — voice %s on %s",
                 settings.kokoro_voice, active[0] if active else "unknown")

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
        """Stream audio out as Kokoro renders it.

        `create_stream` matters more than it looks. `create()` returns only
        when the whole utterance is synthesised — measured at ~1.5 s for a
        3.3 s sentence on CPU, all of it added to time-to-first-audio.
        `create_stream` yields each internal batch as it finishes, so playback
        starts on the first batch and the rest renders underneath it. Same
        total work, a fraction of the perceived latency.
        """
        if self._kokoro is None:
            await self.load()

        # Serialise: concurrent ONNX sessions on one model contend for the same
        # buffers and finish slower than running back to back.
        async with self._lock:
            async for samples, rate in self._kokoro.create_stream(
                text, voice=settings.kokoro_voice, speed=1.0, lang="en-us"
            ):
                audio = np.asarray(samples, dtype=np.float32)
                if rate != self.sample_rate:
                    audio = resample(audio, rate, self.sample_rate)

                # Re-slice to ~120 ms so the browser can begin playing before
                # a whole batch has crossed the socket, and so barge-in cuts
                # cleanly on a small boundary.
                slice_len = int(self.sample_rate * 0.12)
                for start in range(0, len(audio), slice_len):
                    yield audio[start:start + slice_len]

    async def unload(self) -> None:
        self._kokoro = None


def build_local_engine(
    gpu_label: str | None = None,
    ollama_model: str | None = None,
) -> VoiceEngine:
    label = f"Local Engine ({gpu_label})" if gpu_label else "Local Engine"
    return VoiceEngine(
        kind="local",
        label=label,
        stt=FasterWhisperSTT(),
        llm=OllamaLLM(model=ollama_model),
        tts=KokoroTTS(),
    )
