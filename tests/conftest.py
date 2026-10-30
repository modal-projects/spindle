"""Tests read the Miles commit and deployment configs from the environment."""

import gc
import json
import os

import pytest

from spindle.control_plane.deployments import FrontendSettings
from spindle.deployments import config_path, load

os.environ.setdefault("SPINDLE_MILES_COMMIT", "a" * 40)
os.environ.setdefault(
    "SPINDLE_FRONTEND_SETTINGS",
    json.dumps(
        FrontendSettings.from_recipe(load(config_path("qwen35-9b-lora-16k"))).__dict__
    ),
)


@pytest.fixture
def sdk_garbage_collected_off_loop():
    """Keep the cyclic garbage collector off the Tinker SDK's event loop thread."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
