"""Startup phase timing for trainer and sampler containers.

Every container prints one ``spindle_startup_mark`` JSON line per phase boundary,
measured from the moment its process started, so ``modal app logs`` shows where a
cold start spends its time without any extra configuration.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

EVENT_NAME = "spindle_startup_mark"
TRAINER_COMPONENT = "trainer"
BACKEND_COMPONENT = "trainer-backend"
SAMPLER_COMPONENT = "sampler"

_IMPORT_TIME = time.time()


def process_start_time() -> float:
    """Wall-clock time this process started, or the import time if unknown."""
    try:
        with open("/proc/self/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/stat") as handle:
            boot = next(
                int(line.split()[1]) for line in handle if line.startswith("btime")
            )
        return boot + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 - timing must never break startup
        return _IMPORT_TIME


class StartupTimeline:
    """Records named phase boundaries and prints each as a JSON line."""

    def __init__(
        self,
        component: str,
        *,
        origin: float | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        self.component = component
        self.origin = process_start_time() if origin is None else origin
        self.labels = dict(labels or {})
        self.marks: list[tuple[str, float]] = []
        self._last = self.origin

    def mark(self, name: str, **fields: Any) -> float:
        now = time.time()
        phase = now - self._last
        self._last = now
        self.marks.append((name, now))
        print(
            json.dumps(
                {
                    "event": EVENT_NAME,
                    "component": self.component,
                    "mark": name,
                    "ts": round(now, 3),
                    "since_start_s": round(now - self.origin, 3),
                    "phase_s": round(phase, 3),
                    **self.labels,
                    **fields,
                }
            ),
            flush=True,
        )
        return phase

    def phases(self) -> dict[str, float]:
        """Seconds spent before each mark, keyed by the mark that ends the phase."""
        result = {}
        previous = self.origin
        for name, stamp in self.marks:
            result[name] = round(stamp - previous, 3)
            previous = stamp
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "origin": round(self.origin, 3),
            "marks": {name: round(stamp, 3) for name, stamp in self.marks},
            "phases": self.phases(),
        }


_TIMELINES: dict[str, StartupTimeline] = {}


def timeline(component: str, **labels: Any) -> StartupTimeline:
    """The process-wide timeline for ``component``, created on first use."""
    existing = _TIMELINES.get(component)
    if existing is None:
        existing = _TIMELINES[component] = StartupTimeline(component, labels=labels)
    return existing
