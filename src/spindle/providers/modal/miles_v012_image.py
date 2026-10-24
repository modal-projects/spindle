"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"
# A commit, not the branch: Modal caches the fetch layer, so a branch can stay stale.
MILES_COMMIT = "b0c5721131f707de560c600ec6fa110f7ea273db"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
    # Fails the build if the fork branch did not replace the release checkout.
    f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
).add_local_python_source("spindle", ignore=ignore_config_source)
