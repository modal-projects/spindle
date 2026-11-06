import asyncio
import os
from types import SimpleNamespace

import pytest
import tinker
from tinker import types

import spindle
from spindle import tinker_compat
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from tests.support import TinkerStubExecutor, TinkerStubSampler, serve

BASE_MODEL = "Qwen/Qwen3-8B"
DEFINITION = "qwen3_8b"
API_KEY = "tml-spindle-test"
HOSTED_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"


@pytest.fixture(scope="module")
def server_url():
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, TinkerStubExecutor),
        sampling_tasks=LocalSamplingTaskPlatform(TinkerStubSampler()),
    )
    definitions = (
        SimpleNamespace(
            definition_id=DEFINITION,
            name=DEFINITION,
            model=BASE_MODEL,
            parameterization="lora",
            max_context_length=32_768,
        ),
    )
    app = create_control_plane_app(
        plane, definitions, api_key=API_KEY, retrieve_window=5.0
    )
    with serve(app) as url:
        yield url


@pytest.fixture(autouse=True)
def hosted_tinker_env(monkeypatch):
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.delenv("SPINDLE_BASE_URL", raising=False)
    monkeypatch.delenv("SPINDLE_API_KEY", raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "tml-hosted-key")


def train_one_step(base_model: str = BASE_MODEL, **client_kwargs) -> float:
    """Stand-in for an unmodified Tinker script."""
    service = tinker.ServiceClient(**client_kwargs)
    training = service.create_lora_training_client(base_model=base_model, rank=32)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
    )
    result = training.forward_backward([datum], "cross_entropy").result(timeout=30)
    training.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=30)
    return result.metrics["loss:sum"]


def test_decorator_redirects_unmodified_script(server_url):
    @spindle.on_modal(base_url=server_url, api_key=API_KEY)
    def main():
        return train_one_step()

    assert main() == 1.25
    assert os.environ["TINKER_API_KEY"] == "tml-hosted-key"
    assert "TINKER_BASE_URL" not in os.environ


def test_explicit_hosted_base_url_is_redirected(server_url):
    with spindle.on_modal(base_url=server_url, api_key=API_KEY) as target:
        assert target.base_url == server_url
        assert os.environ["TINKER_BASE_URL"] == server_url
        assert train_one_step(base_url=HOSTED_URL, api_key="tml-hosted-key") == 1.25


def test_async_function(server_url):
    @spindle.on_modal(base_url=server_url, api_key=API_KEY)
    async def main():
        service = tinker.ServiceClient()
        capabilities = await service.get_server_capabilities_async()
        return [m.model_name for m in capabilities.supported_models]

    assert asyncio.run(main()) == [BASE_MODEL]


def test_model_map(server_url):
    with spindle.on_modal(
        base_url=server_url,
        api_key=API_KEY,
        model_map={"meta-llama/Llama-3.1-8B": BASE_MODEL},
    ):
        assert train_one_step("meta-llama/Llama-3.1-8B") == 1.25


def test_unsupported_model_lists_supported(server_url):
    with spindle.on_modal(base_url=server_url, api_key=API_KEY):
        with pytest.raises(ValueError, match="Qwen/Qwen3-8B"):
            train_one_step("deepseek-ai/DeepSeek-V3.1")


def test_patch_is_removed_after_exit(server_url):
    original = tinker.ServiceClient.create_lora_training_client
    undo = spindle.patch_tinker(base_url=server_url, api_key=API_KEY)
    assert tinker.ServiceClient.create_lora_training_client is not original
    undo()
    assert tinker.ServiceClient.create_lora_training_client is original
    assert tinker_compat._current() is None


def test_resolves_deployed_server_without_leaking_hosted_key(monkeypatch):
    lookups = []
    monkeypatch.setattr(
        tinker_compat,
        "lookup_server_url",
        lambda app, env: lookups.append(("url", app, env)) or "https://x.modal.run/",
    )
    monkeypatch.setattr(
        tinker_compat,
        "lookup_api_key",
        lambda secret, env: lookups.append(("key", secret, env)) or "tml-spindle",
    )
    target = tinker_compat.resolve_target(app_name="my-spindle", environment="dev")
    assert target == tinker_compat.Target("https://x.modal.run", "tml-spindle")
    assert lookups == [("url", "my-spindle", "dev"), ("key", "spindle-api", "dev")]


def test_reuses_env_when_it_already_points_at_spindle(monkeypatch):
    monkeypatch.setenv("TINKER_BASE_URL", "https://x.modal.run")
    monkeypatch.setenv("TINKER_API_KEY", "tml-spindle")
    target = tinker_compat.resolve_target()
    assert target == tinker_compat.Target("https://x.modal.run", "tml-spindle")
