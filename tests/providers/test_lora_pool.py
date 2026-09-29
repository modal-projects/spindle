import subprocess

import pytest

from spindle.providers.modal import lora_pool
from spindle.providers.modal.lora_pool import LoraPoolSpec


def test_lora_pool_is_shared_by_every_adapter_for_definition() -> None:
    first = LoraPoolSpec("qwen35-9b-lora-16k")
    second = LoraPoolSpec.from_dict(first.as_dict())

    assert first == second
    assert first.app_name == second.app_name
    assert first.app_name == "spindle-lora-qwen35-9b-lora-16k"
    assert first.env() == {
        "SPINDLE_LORA_POOL_APP_NAME": first.app_name,
        "SPINDLE_LORA_POOL_DEFINITION_ID": first.definition_id,
    }


def test_stop_already_stopped_lora_pool_succeeds_but_real_failure_propagates(
    monkeypatch,
):
    monkeypatch.setattr(lora_pool.shutil, "which", lambda _: "/bin/modal")
    result = subprocess.CompletedProcess(
        [], 1, "", "App is already stopped. (Stopped yesterday).\n"
    )
    monkeypatch.setattr(lora_pool.subprocess, "run", lambda *args, **kwargs: result)
    spec = LoraPoolSpec("qwen35-9b-lora-16k")
    lora_pool.stop_pool(spec)
    result.stderr = "authentication failed"
    with pytest.raises(subprocess.CalledProcessError):
        lora_pool.stop_pool(spec)
