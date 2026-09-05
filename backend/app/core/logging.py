"""Logging setup.

Timing numbers are the whole point of the console output here, so the format
keeps the logger name short and puts the message where you can scan a column of
turn latencies without your eye jumping.
"""

from __future__ import annotations

import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # These are chatty at DEBUG and drown out the pipeline timings.
    for noisy in ("httpx", "httpcore", "groq", "urllib3", "asyncio",
                  "python_multipart", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Uvicorn's own access log duplicates what we already print per turn.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
