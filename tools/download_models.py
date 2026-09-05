#!/usr/bin/env python3
"""Fetch the model weights EchoSync needs on disk.

Two files are not pip-installable and must be downloaded once:

* **Silero VAD** (~1.8 MB) — required by *every* configuration, local or cloud,
  because segmentation always happens on our side.
* **Kokoro TTS** (~330 MB model + ~27 MB voices) — required unless you set
  ``CLOUD_TTS_PROVIDER=edge``.

Ollama models are deliberately not handled here: `ollama pull` already does
this properly, with its own resume and dedup.

    python tools/download_models.py            # everything needed
    python tools/download_models.py --vad-only # skip the 330 MB download
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"

ASSETS: dict[str, tuple[str, str]] = {
    "silero_vad.onnx": (
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "Silero VAD",
    ),
    "kokoro-v0_19.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "Kokoro TTS model",
    ),
    "voices.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin",
        "Kokoro voices",
    ),
}

VAD_ONLY = {"silero_vad.onnx"}


def _progress(done: int, total: int, label: str) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {label}: {done / 1e6:.1f} MB")
    else:
        pct = 100.0 * done / total
        bar = "█" * int(pct // 3) + "░" * (33 - int(pct // 3))
        sys.stdout.write(f"\r  {label}: {bar} {pct:5.1f}%  {done / 1e6:6.1f}/{total / 1e6:.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: Path, label: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {label}: already present ({dest.stat().st_size / 1e6:.1f} MB) — skipping")
        return

    # Download to a temp name and rename on success, so an interrupted run
    # never leaves a truncated file that looks valid to the loader.
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while chunk := response.read(1 << 16):
                fh.write(chunk)
                done += len(chunk)
                _progress(done, total, label)
        print()
        tmp.rename(dest)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"\n  ✗ {label} failed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Download EchoSync model weights.")
    parser.add_argument("--vad-only", action="store_true",
                        help="Only fetch Silero VAD (use with CLOUD_TTS_PROVIDER=edge).")
    parser.add_argument("--dest", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    wanted = VAD_ONLY if args.vad_only else set(ASSETS)

    print(f"Downloading into {args.dest}\n")
    for name, (url, label) in ASSETS.items():
        if name not in wanted:
            continue
        download(url, args.dest / name, label)

    print("\n✓ Models ready.")
    if not args.vad_only:
        print("\nNext: pull the local LLM (skip if you are cloud-only):")
        print("  ollama pull llama3.2:3b-instruct-q4_K_M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
