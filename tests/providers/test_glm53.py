import argparse

from copy import deepcopy
from types import SimpleNamespace

import pytest

from spindle.backends.miles_config import lora_target_flags
from spindle.backends.miles_arguments import configure_parser
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal import deployment_apps


def test_glm_recipe_survives_worker_serialization():
    config = DeploymentConfig.create(load(config_path("glm53-flash-lora-16k")))
    worker = DeploymentConfig.model_validate_json(config.model_dump_json())
    assert worker.trainer_identity() == config.trainer_identity()
    assert worker.inference_identity() == config.inference_identity()
    targets = worker.inference_settings["lora_target_modules"]
    # The absorbed KV-up path in the upstream provider supports only single LoRA.
    assert "kv_b_proj" not in targets
    assert {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"} <= set(targets)
    assert lora_target_flags(tuple(targets)) == (True, True, False)
    assert worker.trainer_settings["miles"]["context_parallel_size"] == 1


@pytest.mark.parametrize(
    "field,identity",
    [("trainer_image", "trainer_identity"), ("inference_image", "inference_identity")],
)
def test_image_change_redeploys_its_app(field, identity):
    before = DeploymentConfig.create(load(config_path("qwen35-9b-lora-16k")))
    after = deepcopy(before)
    setattr(after.recipe, field, "custom.images:runtime")
    assert getattr(before, identity)() != getattr(after, identity)()
    other = (
        "inference_identity" if identity == "trainer_identity" else "trainer_identity"
    )
    assert getattr(before, other)() == getattr(after, other)()


def test_custom_image_is_resolved_on_deployer_only(monkeypatch):
    selected = object()
    imports = []

    def import_module(name):
        imports.append(name)
        return SimpleNamespace(trainer_image=selected)

    monkeypatch.setattr(deployment_apps.importlib, "import_module", import_module)
    monkeypatch.setattr(deployment_apps.modal, "is_local", lambda: True)
    assert deployment_apps.image_for("miles", "custom.images:trainer_image") is selected
    assert imports == ["custom.images"]
    monkeypatch.setattr(deployment_apps.modal, "is_local", lambda: False)
    deployment_apps.image_for("miles", "custom.images:trainer_image")
    assert imports == ["custom.images"]


@pytest.mark.parametrize(
    "target",
    [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "b_proj",
        "f_a_proj",
        "f_b_proj",
        "g_a_proj",
        "g_b_proj",
    ],
)
def test_mla_and_kda_targets_report_attention_training(target):
    assert lora_target_flags((f"model.layers.*.self_attn.{target}",)) == (
        True,
        False,
        False,
    )


def test_miles_overrides_wait_for_late_registered_arguments():
    parser = argparse.ArgumentParser()
    configure_parser(parser, {"dsa_attention_backend": "tilelang", "feature": False})
    # Miles calls add_custom_arguments before registering these native options.
    parser.add_argument("--dsa-attention-backend", choices=["megatron", "tilelang"])
    parser.add_argument("--feature", action="store_true", default=True)
    args = parser.parse_args(["--dsa-attention-backend", "megatron", "--feature"])
    assert args.dsa_attention_backend == "tilelang"
    assert args.feature is False


def test_miles_deferred_overrides_preserve_native_validation():
    parser = argparse.ArgumentParser()
    configure_parser(parser, {"backend": "invalid"})
    parser.add_argument("--backend", choices=["valid"])
    with pytest.raises(SystemExit):
        parser.parse_args([])
