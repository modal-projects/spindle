"""Build the small Python extension once per Rust source revision."""

from pathlib import Path
import modal

SOURCE = Path(__file__).resolve().parents[4] / "rust/router"


def router_image():
    return (
        modal.Image.from_registry("rust:1.90-slim-bookworm", add_python="3.12")
        .apt_install("git")
        .pip_install("maturin==1.15.0")
        .add_local_dir(
            SOURCE, "/opt/spindle-router", copy=True, ignore=["target", "dist", ".venv"]
        )
        .run_commands(
            "cd /opt/spindle-router && maturin build --release --locked --out /tmp/router-wheels",
            "pip install /tmp/router-wheels/*.whl",
        )
    )
