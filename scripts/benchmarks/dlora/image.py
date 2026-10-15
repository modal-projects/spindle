"""Pinned merge-PR engine with the existing Stitch/Spindle LoRA sidecar."""

from pathlib import Path

from spindle.providers.modal.merge_image import base_image

REVISION = "708d38f143697e25c3e687c20fc0b043930c8416"
ROOT = Path(__file__).resolve().parents[3]


def measured_image():
    return (
        base_image.add_local_dir(
            ROOT / "src/spindle",
            "/opt/dlora-spindle/spindle",
            copy=True,
            ignore=lambda p: p.suffix != ".py",
        )
        .add_local_file(
            Path(__file__).with_name("instrument.py"),
            "/opt/dlora-instrument.py",
            copy=True,
        )
        .add_local_file(
            ROOT / "scripts/benchmarks/lora_routing/cache_instrumentation.py",
            "/opt/cache_instrumentation.py",
            copy=True,
        )
        .run_commands("python /opt/dlora-instrument.py")
    )
