"""Tests read the Miles commit and deployment configs from the environment."""

import gc
import json
import os
import tempfile

import pytest

from spindle.deployments import DeploymentConfig, config_path, load

os.environ.setdefault("SPINDLE_MILES_COMMIT", "a" * 40)
# Importing spindle.providers.modal.app needs explicit offline configs.
if "SPINDLE_DEPLOYMENT_CONFIGS_FILE" not in os.environ:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="spindle-test-configs-", delete=False
    ) as configs_file:
        json.dump(
            [
                DeploymentConfig.create(load(config_path(name))).model_dump(mode="json")
                for name in ("qwen35-9b-fft-64k", "qwen35-9b-lora-16k")
            ],
            configs_file,
        )
    os.environ["SPINDLE_DEPLOYMENT_CONFIGS_FILE"] = configs_file.name


@pytest.fixture
def sdk_garbage_collected_off_loop():
    """Keep the cyclic garbage collector off the Tinker SDK's event loop thread."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
