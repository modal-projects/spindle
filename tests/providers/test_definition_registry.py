import asyncio
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.local import InMemoryKeyValueStore
from spindle.providers.modal import app as modal_app
from tests.support import RoutedTestStore

from .test_modal_app import register_recipes


def test_definition_registry_resolves_every_definition(monkeypatch) -> None:
    definitions = [
        DeploymentConfig.create(load(config_path(name)))
        for name in ("qwen35-9b-fft-64k", "qwen35-9b-lora-16k")
    ]
    store = InMemoryKeyValueStore()
    monkeypatch.setattr(modal_app, "deployment_kv", lambda: store)
    shared = InMemoryKeyValueStore()
    monkeypatch.setattr(modal_app, "shared_kv", lambda: RoutedTestStore(shared))
    monkeypatch.setattr(modal_app.DEPLOYMENTS, "_version_store", None)
    register_recipes(modal_app, store, *definitions)
    for definition in definitions:
        assert (
            asyncio.run(modal_app.module_for(definition.name)).name == definition.name
        )
        assert asyncio.run(modal_app.parameterization_for(definition.name)) == (
            definition.parameterization
        )
        assert definition.trainer_app_name
