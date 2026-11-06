from copy import deepcopy
from types import SimpleNamespace

from spindle.configs.qwen35_9b_lora_16k import Config
from spindle.deployments import DeploymentConfig
from spindle.providers.modal import deployment_apps


def test_custom_sampler_image_changes_deployment_identity():
    before = DeploymentConfig.create(Config())
    after = deepcopy(before)
    after.recipe.inference_image = "custom.images:inference"
    assert after.inference_identity() != before.inference_identity()
    assert after.trainer_identity() == before.trainer_identity()


def test_custom_image_is_resolved_on_deployer_only(monkeypatch):
    selected = object()
    imported = []

    def import_module(name):
        imported.append(name)
        return SimpleNamespace(inference=selected)

    monkeypatch.setattr(deployment_apps.importlib, "import_module", import_module)
    monkeypatch.setattr(deployment_apps.modal, "is_local", lambda: True)
    assert deployment_apps.image_for("sglang", "custom.images:inference") is selected
    monkeypatch.setattr(deployment_apps.modal, "is_local", lambda: False)
    deployment_apps.image_for("sglang", "custom.images:inference")
    assert imported == ["custom.images"]


def test_custom_sampler_images_are_built_once_each(monkeypatch):
    built = []
    app = object()
    monkeypatch.setattr(deployment_apps.modal.App, "lookup", lambda *a, **k: app)
    monkeypatch.setattr(
        deployment_apps,
        "image_for",
        lambda backend, name: SimpleNamespace(
            build=lambda target: built.append((name, target))
        ),
    )
    deployment_apps.build_sampler_image(
        "example", "dev", ("", "custom:inference", "custom:inference")
    )
    assert built == [("", app), ("custom:inference", app)]
