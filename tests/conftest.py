"""Tests read the Miles commit and deployment configs from the environment."""

import gc
import json
import os

import pytest

from spindle.deployments import DeploymentConfig, config_path, load

os.environ.setdefault("SPINDLE_MILES_COMMIT", "a" * 40)
# Importing spindle.providers.modal.app needs explicit offline configs.
os.environ.setdefault(
    "SPINDLE_DEPLOYMENT_CONFIGS",
    json.dumps(
        [
            DeploymentConfig.create(load(config_path(name))).model_dump(mode="json")
            for name in ("qwen35-9b-fft-64k", "qwen35-9b-lora-16k")
        ]
    ),
)


@pytest.fixture
def sdk_garbage_collected_off_loop():
    """Keep the cyclic garbage collector off the Tinker SDK's event loop thread.

    Every Tinker client in the process shares one background event loop, and
    tests never close their clients. When a collection runs on that loop's
    thread, an abandoned HTTP client's finalizer schedules aclose() for sockets
    the collector has already closed. That close then unregisters the fd from
    the selector after a new connection has reused the fd, so the new connect
    never wakes up and the request stalls for the SDK's connect timeout and
    backoff. A collection on this thread has no running loop, so the finalizers
    schedule nothing.
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
