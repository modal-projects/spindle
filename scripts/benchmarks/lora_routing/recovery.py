"""Durable update commits and conservative retry classification for RL clients."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def latest_commit(root):
    paths = sorted(root.glob("commit-*.json"))
    if not paths:
        return None
    commits = [json.loads(path.read_text()) for path in paths]
    if [row["update"] for row in commits] != list(range(1, len(commits) + 1)):
        raise ValueError("Noncontiguous committed training updates")
    return commits[-1]


def classify_failure(message):
    message = message.lower()
    if any(
        word in message
        for word in (
            "out of memory",
            "cuda error",
            "nonfinite",
            "optimizer update was skipped",
            "invalid sampled",
            "mismatch",
            "assertionerror",
            "valueerror",
            "typeerror",
            "keyerror",
            "attributeerror",
            "syntaxerror",
            "permission",
            "unauthorized",
        )
    ):
        return "requires_fix"
    if any(
        word in message
        for word in (
            "timeout",
            "timed out",
            "connection",
            "unavailable",
            "503",
            "502",
            "504",
            "429",
            "500",
            "backendfailed",
            "backend failed",
            "trainer exited",
            "worker died",
            "actor died",
            "not found",
            "terminated",
            "cancelled",
            "canceled",
            "session expired",
            "model is failed",
            "model has failed",
            "model failed",
            "model state",
            "engine",
            "failed model",
        )
    ):
        return "retry_from_checkpoint"
    return "requires_investigation"


class ExistingProcess:
    """Observe children retained by the paused old coordinator during handover."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        try:
            fields = (
                Path(f"/proc/{self.pid}/stat").read_text().rsplit(") ", 1)[1].split()
            )
        except FileNotFoundError:
            return 1  # Losing exit evidence must not report successful completion.
        if fields[0] == "Z":
            return os.waitstatus_to_exitcode(int(fields[49]))
        return None

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(1)
        return self.poll()
