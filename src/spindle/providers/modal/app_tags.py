"""App attribution shared by local launches and remote pool provisioners."""

import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

REVISION_ENV = "SPINDLE_REVISION"


def spindle_revision() -> str:
    if revision := os.environ.get(REVISION_ENV):
        return revision

    # Resolve relative to this package, never the launching application's cwd.
    root = Path(__file__).resolve().parents[4]
    if (root / ".git").exists():
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass

    # pip/uv record the commit when installing directly from a Git URL.
    try:
        direct_url = distribution("modal-spindle").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    if direct_url:
        revision = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if revision:
            return revision
    return "unknown"


def app_tags(role: str) -> dict[str, str]:
    return {
        "_modal_source": "spindle",
        "spindle_rev": spindle_revision(),
        "spindle_role": role,
    }


def revision_env() -> dict[str, str]:
    """Carry source identity into containers where the Git checkout is absent."""
    return {REVISION_ENV: spindle_revision()}
