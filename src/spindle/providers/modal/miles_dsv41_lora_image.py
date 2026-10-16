"""Trainer image for DeepSeek-V4.1 multi-LoRA through Miles.

The v0.1.2 release image with Miles replaced by a branch that adds
multi-LoRA to DeepSeek-V4.1's raw (non-Bridge) model path. Point MILES_REF at
a newer branch or commit to update it, then refresh the trainer app.
"""

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_REF = "1f4fba21d9b4483ceedade9c2e3954ed1f670768"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_REF}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    f"test -f {MILES_PATH}/miles/backends/megatron_utils/lora/actor.py",
).add_local_python_source("spindle", ignore=ignore_config_source)
