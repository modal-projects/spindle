"""Identify the Spindle release and source commit a deployment runs."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from spindle import __version__

COMMIT_ENV = "SPINDLE_COMMIT"


def source_commit() -> str | None:
    """Deployed containers read the commit recorded at deploy time."""
    return os.environ.get(COMMIT_ENV) or _local_commit()


@cache
def _local_commit() -> str | None:
    try:
        direct_url = distribution("modal-spindle").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def version_tags() -> dict[str, str]:
    return {
        "spindle_version": __version__,
        "spindle_commit": source_commit() or "unknown",
    }
