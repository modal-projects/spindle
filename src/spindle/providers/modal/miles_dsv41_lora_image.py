"""Miles v0.1.2 release image with multi-LoRA for DeepSeek-V4.1."""

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_BRANCH}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
).add_local_python_source("spindle", ignore=ignore_config_source)
