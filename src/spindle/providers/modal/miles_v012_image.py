"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"
# Head of MILES_BRANCH. Modal caches this layer by command, so fetching the branch
# by name would keep serving an older head.
MILES_COMMIT = "11286edcb5a74b6bc48187ce7ada563f6a84fc4c"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
    # Fails the build if the fork branch did not replace the release checkout.
    f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
).add_local_python_source("spindle", ignore=ignore_config_source)
