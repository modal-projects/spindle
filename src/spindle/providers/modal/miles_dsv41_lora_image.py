"""Trainer image for DeepSeek-V4.1 multi-LoRA through Miles.

The v0.1.2 release image with Miles replaced by a branch that adds
multi-LoRA to DeepSeek-V4.1's raw (non-Bridge) model path. Point MILES_REF at
a newer branch or commit to update it, then refresh the trainer app.
"""

from .image_dependencies import ignore_config_source
from .miles_full_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_REF = "db783526e5cbdda917da18657220cba10a5f73a4"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_REF}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    'python -c "from miles.backends.megatron_utils.lora.actor import '
    "MultiLoRATrainRayActor; from miles.tinker.runtime import MilesBackend as B; "
    "assert all(hasattr(B, name) for name in "
    "('load_slot', 'unload_slot', 'forward_backward', "
    "'forward_only', 'optim_step', 'save_slot', 'export_slot'))\"",
).add_local_python_source("spindle", ignore=ignore_config_source)
