from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping

REQUEST_TIMING_ENV = "SPINDLE_REQUEST_TIMING"
EVENT_NAME = "spindle_request_mark"

_ENABLED = os.environ.get(REQUEST_TIMING_ENV) == "1"


def enabled(env: Mapping[str, str] = os.environ) -> bool:
    return env.get(REQUEST_TIMING_ENV) == "1"


def mark(name: str, **fields: object) -> None:
    """Emit one ``spindle_request_mark`` JSON line; no-op unless SPINDLE_REQUEST_TIMING=1."""
    if not _ENABLED:
        return
    print(
        json.dumps({"event": EVENT_NAME, "mark": name, "ts": time.time(), **fields}),
        flush=True,
    )
