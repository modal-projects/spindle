"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

from .image_dependencies import ignore_config_source
from .bridge_import_cache import BRIDGE_IMPORT_CACHE_PATCH
from .miles_release_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "codex/glm53-shared-v012"
# Head of MILES_BRANCH. Modal caches this layer by command, so fetching the branch
# by name would keep serving an older head.
MILES_COMMIT = "953c7cc112dda063ba69782c5aa21e3a1444925c"

image = (
    release_image.pip_install("transformers==5.16.0", "tilelang==0.1.12")
    .run_commands(
        f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
        f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
        # Fails the build if the fork branch did not replace the release checkout.
        f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
        # Grouped imports otherwise retain every expanded expert tensor on each
        # rank. This shared loader fix also applies to non-GLM MoE checkpoints.
        "python - <<'PY'\n"
        "from importlib.metadata import distribution\n"
        "import subprocess\n"
        "root = distribution('megatron-bridge').locate_file('')\n"
        "subprocess.run(['patch', '-p1', '-d', str(root)], input="
        + repr(BRIDGE_IMPORT_CACHE_PATCH)
        + ", text=True, check=True)\nPY\n",
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
