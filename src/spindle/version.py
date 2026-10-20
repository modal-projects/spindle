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
    source = Path(__file__)
    try:
        # A tracked version.py means a Spindle checkout, not a project that vendors an install.
        _git(source.parent, "ls-files", "--error-unmatch", source.name)
        return _git(source.parent, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def version_tags() -> dict[str, str]:
    commit = source_commit()
    return {
        "spindle_version": __version__,
        "spindle_commit": commit[:7] if commit else "unknown",
    }
