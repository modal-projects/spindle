"""Add the router extension to an image, leaving no build toolchain behind."""

import sysconfig
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "rust/router"
if not SOURCE.is_dir():
    SOURCE = Path(sysconfig.get_path("data")) / "share/spindle/router"

# One shell command, so the toolchain never persists in an image layer.
BUILD = " && ".join(
    [
        "apt-get update",
        "apt-get install -y --no-install-recommends curl ca-certificates gcc libc6-dev",
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.90.0",
        "pip install maturin==1.15.0",
        "cd /opt/spindle-router",
        "PATH=/root/.cargo/bin:$PATH maturin build --release --locked --out /tmp/router-wheels",
        "pip install /tmp/router-wheels/*.whl",
        "pip uninstall -y maturin",
        "cd /",
        "rm -rf /root/.rustup /root/.cargo /opt/spindle-router /tmp/router-wheels",
        "apt-get purge -y --auto-remove curl gcc libc6-dev",
        "rm -rf /var/lib/apt/lists/*",
    ]
)


def with_router(image):
    return image.add_local_dir(
        SOURCE, "/opt/spindle-router", copy=True, ignore=["target", "dist", ".venv"]
    ).run_commands(BUILD)
