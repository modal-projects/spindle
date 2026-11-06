import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from spindle.backends.deployment import backend_config, serving_options
from spindle.backends.megatron_config import parse_backend_config
from spindle.backends.miles_config import MilesBackendConfig, lora_target_flags
from spindle.backends.miles_arguments import (
    apply_config_overrides,
    defer_config_overrides,
)
from spindle.configs.qwen35_9b_lora_16k import Config as Parent
from spindle.configuration import BaseConfig
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.control_plane.deployments import (
    DeploymentConflict,
    DeploymentRequest,
    DeploymentRoutes,
)
from spindle.control_plane.keys import model_key, placement_key
from spindle.control_plane.records import ModelRecord
from spindle.deployments import (
    DeploymentConfig,
    config_path,
    load,
    validate_frontend,
)
from spindle.providers.local import InMemoryKeyValueStore
from spindle.providers.modal.deployment_apps import deployment_env


def recipe(preset="qwen35-9b-lora-16k", **changes):
    data = vars(load(config_path(preset)))
    for path, value in changes.items():
        keys = path.split("__")
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return BaseConfig(**data)


def resolved(spec=None, **changes):
    return DeploymentConfig.create(spec or recipe(**changes))


class CountingStore(InMemoryKeyValueStore):
    """An in-memory registry that counts full reads and single-file lookups."""

    def __init__(self, values=None):
        super().__init__(values)
        self.reads = 0
        self.gets = 0

    async def list_keys(self, prefix):
        self.reads += 1
        return await super().list_keys(prefix)

    async def get(self, key):
        self.gets += 1
        return await super().get(key)


def routes_for(*deployments):
    kv = CountingStore({d.name: d.model_dump(mode="json") for d in deployments})
    return DeploymentRoutes(lambda: kv), kv


@pytest.mark.parametrize("preset", ["qwen35-9b-lora-16k", "qwen35-4b-fft-64k"])
def test_sampler_metrics_are_enabled_by_default_and_can_be_disabled(preset):
    assert resolved(recipe(preset)).inference_settings["enable_metrics"] is True
    disabled = resolved(recipe(preset, sglang_cfg__enable_metrics=False))
    assert disabled.inference_settings["enable_metrics"] is False


def test_multi_node_lora_captures_land_on_the_checkpoint_volume():
    assert "capture_dir" not in backend_config(recipe())
    multi_node = backend_config(recipe("qwen38-27b-lora-256k"))
    assert multi_node["capture_dir"] == "/checkpoints/.spindle-captures"
    dsv41 = backend_config(recipe("deepseek-v41-flash-lora-8k"))
    assert dsv41["capture_dir"] == "/checkpoints/.spindle-captures"


def test_presets_context_topology_and_backend_options():
    spec = recipe()
    config = backend_config(spec)["miles"]
    assert (
        config["actor_num_gpus_per_node"],
        config["tensor_model_parallel_size"],
        config["max_lora_slots"],
    ) == (4, 4, 6)
    assert config["cli_options"]["recompute_num_layers"] == 1
    assert config["extra_args"] == ("--seq-length", "16384")
    large = recipe("qwen35-9b-lora-64k")
    assert large.max_context_length == 65536
    assert backend_config(large)["miles"]["actor_num_gpus_per_node"] == 8
    fft = backend_config(recipe("qwen35-4b-fft-64k"))["megatron"]
    assert (fft["tensor_model_parallel_size"], fft["context_parallel_size"]) == (2, 2)
    assert fft["provider_overrides"]["recompute_granularity"] == "full"


def test_no_model_catalog_required():
    spec = recipe(
        model="my-org/new-model",
        miles_cfg__model_type="",
        miles_cfg__cli_options={
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        },
    )
    assert resolved(spec).model == "my-org/new-model"
    assert backend_config(spec)["miles"]["model_type"] == ""


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"miles_cfg__cli_options": {"hf_checkpoint": "other"}}, "managed"),
        (
            {"miles_cfg__cli_options": {"pipeline_model_parallel_size": 2}},
            "managed",
        ),
        ({"sglang_cfg": {"model_path": "other"}}, "managed"),
        ({"sglang_cfg": {"tp_size": 2}}, "replica GPU"),
        ({"trainer_max_clients_per_instance": 7}, "max_lora_slots"),
        ({"trainer_max_clients_per_batch": 0}, "trainer_max_clients_per_batch"),
    ],
)
def test_invalid_integrations_fail_when_building_backend_settings(changes, match):
    spec = recipe(**changes)
    with pytest.raises(ValueError, match=match):
        backend_config(spec)
        serving_options(spec)


def test_deepseek_v41_lora_recipe():
    row = resolved(recipe("deepseek-v41-flash-lora-8k"))
    miles = MilesBackendConfig(**row.trainer_settings["miles"])
    assert (
        miles.world_size,
        miles.tensor_model_parallel_size,
        miles.context_parallel_size,
        miles.expert_model_parallel_size,
        miles.expert_tensor_parallel_size,
    ) == (16, 1, 1, 16, 1)
    arguments = miles.miles_arguments()
    assert arguments[arguments.index("--megatron-to-hf-mode") + 1] == "raw"
    assert arguments[arguments.index("--max-tokens-per-gpu") + 1] == "8192"
    assert miles.cli_options["moe_router_freeze_gate"] is True
    assert lora_target_flags(miles.target_modules) == (True, True, False)

    serving = row.inference_settings
    assert serving["enable_lora"] is True
    assert serving["max_lora_rank"] == 32
    assert serving["lora_target_modules"] == [
        "attn.wq_a",
        "attn.wkv",
        "attn.wq_b",
        "attn.wo_b",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w2",
        "ffn.shared_experts.w3",
    ]
    assert serving["disable_shared_experts_fusion"] is True
    assert row.recipe.inference_env["SGLANG_DSV4_FP4_DEQUANT"] == "1"
    assert serving["weight_loader_disable_mmap"] is False
    assert row.recipe.trainer_image == "miles-v0.1.2"
    assert row.model == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert row.asset_path == "/assets/deepseek-ai/DeepSeek-V4.1-Flash"
    assert json.loads(serving["json_model_override_args"])["vision_n_layers"] == 0


@pytest.mark.parametrize(
    "name, tp",
    [("deepseek-v41-flash-lora-32k", 4), ("deepseek-v41-flash-lora-32k-b300", 2)],
)
def test_deepseek_v41_lora_32k_recipes_keep_tensor_parallelism(name, tp):
    row = resolved(recipe(name))
    miles = MilesBackendConfig(**row.trainer_settings["miles"])
    assert (miles.tensor_model_parallel_size, miles.expert_model_parallel_size) == (
        tp,
        16,
    )
    assert row.inference_settings["context_length"] == 32768


@pytest.mark.parametrize(
    "name", ["deepseek-v41-flash-lora-64k", "deepseek-v41-flash-lora-128k"]
)
def test_deepseek_v41_long_context_recipes_train_one_row_per_micro_batch(name):
    miles = MilesBackendConfig(**resolved(recipe(name)).trainer_settings["miles"])
    arguments = miles.miles_arguments()
    assert arguments[arguments.index("--micro-batch-size") + 1] == "1"
    assert miles.cli_options["qkv_format"] == "bshd"


@pytest.mark.parametrize("context, tp", [("8k", 1), ("32k", 4)])
def test_deepseek_v41_1017_recipes_load_the_1017_release(context, tp):
    row = resolved(recipe(f"deepseek-v41-flash-1017-lora-{context}"))
    miles = MilesBackendConfig(**row.trainer_settings["miles"])
    assert miles.tensor_model_parallel_size == tp
    assert row.asset_path == "/assets/deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert row.recipe.weights_revision == "1bae0e1088993b9b82de08f59de5be5a34e42170"


def test_asset_paths_follow_the_model():
    a = resolved()
    assert a.asset_path == f"/assets/{a.recipe.model}"
    b = resolved(recipe(model="other/Qwen3.5-9B-Base"))
    assert a.asset_path != b.asset_path


@pytest.mark.parametrize("field", ["trainer_disk_mib", "inference_disk_mib"])
def test_disk_requests_below_modal_minimum_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        resolved(**{field: 100_000})


def test_model_weights_override_assets_but_not_the_served_model():
    row = resolved(recipe(model_weights="org/model-bf16"))
    assert row.model == row.recipe.model
    assert row.weights_repo == "org/model-bf16"
    assert row.asset_path == "/assets/org/model-bf16"
    assert row.trainer_settings["miles"]["hf_checkpoint"] == row.asset_path
    assert row.trainer_identity() != resolved().trainer_identity()
    assert row.inference_identity() != resolved().inference_identity()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (dict(inference_lora_routing="nearest"), "unknown inference_lora_routing"),
        (
            dict(inference_routing_prefill_tokens_per_second=0),
            "prefill_tokens_per_second",
        ),
        *(
            (dict(inference_lora_routing="cache_wait", **change), match)
            for change, match in [
                (dict(sglang_cfg__dp_size=2), "data-parallel"),
                (dict(inference_routing_balance_relative=0.5), "balance_relative"),
                (dict(inference_routing_balance_absolute=-1), "balance_absolute"),
                (dict(inference_routing_cache_threshold=1.5), "cache_threshold"),
            ]
        ),
    ],
)
def test_invalid_routing_fails_at_deploy(changes, match):
    with pytest.raises(ValueError, match=match):
        resolved(**changes)


@pytest.mark.parametrize(
    ("preset", "changes", "routing"),
    [
        ("qwen35-9b-lora-16k", {}, "cache_wait"),
        ("qwen35-9b-lora-16k", dict(sglang_cfg__dp_size=2), "affinity"),
        ("qwen35-4b-fft-64k", {}, "affinity"),
        ("qwen35-9b-lora-16k", dict(inference_lora_routing="affinity"), "affinity"),
    ],
)
def test_lora_pools_with_one_data_parallel_rank_default_to_cache_wait(
    preset, changes, routing
):
    row = resolved(recipe(preset, **changes))
    assert row.recipe.inference_lora_routing == routing


@pytest.mark.parametrize(
    ("running", "absolute"), [(4, 4), (32, 13), (80, 32), (128, 51)]
)
def test_cache_wait_balance_scales_with_running_requests(running, absolute):
    row = resolved(sglang_cfg__max_running_requests=running)
    assert row.recipe.inference_routing_balance_absolute == absolute
    assert row.recipe.inference_routing_prefill_tokens_per_second is None
    configured = resolved(
        sglang_cfg__max_running_requests=running, inference_routing_balance_absolute=7
    )
    assert configured.recipe.inference_routing_balance_absolute == 7


def test_switching_routing_redeploys_the_pool():
    # Only cache_wait pools run the observer.
    affinity = resolved(inference_lora_routing="affinity")
    assert affinity.inference_identity() != resolved().inference_identity()


def test_trainer_image_is_part_of_trainer_identity():
    trainer = resolved(recipe(trainer_image="trainer-x"))
    assert trainer.trainer_identity() != resolved().trainer_identity()
    assert trainer.inference_identity() == resolved().inference_identity()


@pytest.mark.anyio
async def test_routing_prefers_longest_context():
    small = resolved(recipe("deepseek-v41-flash-lora-8k"))
    large = resolved(recipe("deepseek-v41-flash-lora-32k"))
    routes, _ = routes_for(small, large)
    assert (await routes.select(small.model, "lora")).definition_id == large.name
    assert (await routes.select(small.name)).definition_id == small.name
    assert (await routes.capabilities())[0] == {
        "model_name": small.model,
        "max_context_length": 32768,
    }
    validate_frontend([small.recipe, large.recipe])

    twin = resolved(recipe("deepseek-v41-flash-lora-8k", name="twin"))
    routes, _ = routes_for(small, twin)
    assert (await routes.select(small.model, "lora")).definition_id == small.name

    other = resolved(recipe(name="other", model="org/other"))
    routes, _ = routes_for(other)
    assert (await routes.select(other.model, "lora")).definition_id == other.name


@pytest.mark.anyio
async def test_sampling_prefers_longest_context_and_training_filters_parameterization():
    lora = resolved()
    fft = resolved(recipe("qwen35-4b-fft-64k", model=lora.model))
    for deployments in ((lora, fft), (fft, lora)):
        routes, _ = routes_for(*deployments)
        assert (await routes.select(lora.model)).name == fft.name
        assert (await routes.select(lora.model, "lora")).name == lora.name
        assert (await routes.select(lora.model, "full")).name == fft.name
        assert (await routes.select(deployments[1].name)).name == deployments[1].name
        assert await routes.select("missing") is None
        assert await routes.select(lora.name, "full") is None


def test_native_false_list_aliases_and_scalar_overrides():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-feature", action="store_true")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--tp", "--tensor-parallel-size", dest="tp_size", type=int)
    parser.add_argument("--unchanged")
    argv = ["--use-feature", "--layers", "1", "2", "--tp=4", "--unchanged", "keep"]
    apply_config_overrides(
        parser, {"use_feature": False, "layers": [3], "tp_size": 8}, argv
    )
    parsed = parser.parse_args(argv)
    assert vars(parsed) == {
        "use_feature": False,
        "layers": [3],
        "tp_size": 8,
        "unchanged": "keep",
    }
    with pytest.raises(ValueError, match="unknown backend option"):
        apply_config_overrides(parser, {"typo": 1}, [])
    with pytest.raises(ValueError, match="boolean"):
        apply_config_overrides(parser, {"use_feature": "false"}, [])
    with pytest.raises(ValueError, match="list"):
        apply_config_overrides(parser, {"layers": "1,2"}, [])


def test_deferred_overrides_see_arguments_added_after_hook():
    parser = argparse.ArgumentParser()
    argv = ["--qkv-format", "bshd"]
    defer_config_overrides(parser, {"qkv_format": "thd", "fused": False}, argv)
    parser.add_argument("--qkv-format")
    parser.add_argument("--fused", action="store_true")
    assert vars(parser.parse_args(argv)) == {"qkv_format": "thd", "fused": False}
    assert vars(parser.parse_args(argv)) == {"qkv_format": "thd", "fused": False}
    assert argv == ["--qkv-format", "bshd", "--qkv-format=thd"]


@pytest.mark.anyio
async def test_multiple_models_same_http_service_and_old_binding_survives_switch():
    store = InMemoryKeyValueStore()
    plane = ControlPlane(store, SimpleNamespace())
    first = resolved()
    other = resolved(recipe(name="other", model="org/other-model"))
    app = create_control_plane_app(plane, [first, other], api_key="test")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://one-frontend",
        headers={"x-api-key": "test"},
    ) as client:
        session = (await client.post("/api/v1/create_session", json={})).json()[
            "session_id"
        ]
        for seq, row in enumerate([first, other]):
            response = await client.post(
                "/api/v1/create_model",
                json={
                    "session_id": session,
                    "model_seq_id": seq,
                    "base_model": row.model,
                    "lora_config": {"rank": 32},
                },
            )
            assert response.status_code == 200, response.text
            record = await store.get(model_key(response.json()["model_id"]))
            assert record["engine_definition_id"] == row.definition_id
        configs = await client.get("/api/v1/spindle/deployments")
        assert len(configs.json()["deployments"]) == 2
        client.headers.clear()
        assert (await client.get("/api/v1/spindle/deployments")).status_code == 401


async def names(routes):
    return [d.name for d in await routes.definitions()]


@pytest.mark.anyio
async def test_requests_for_cached_recipes_and_models_never_read_the_registry():
    base = resolved()
    routes, store = routes_for(base)
    await routes.definitions()
    before = (store.reads, store.gets)
    assert (await routes.select(base.model)).name == base.name
    assert (await routes.select(base.name)).name == base.name
    assert await routes.select("Unknown/Model") is None
    assert (store.reads, store.gets) == before


@pytest.mark.anyio
async def test_a_newly_registered_recipe_is_found_on_its_first_request():
    base = resolved()
    routes, store = routes_for(base)
    await routes.definitions()
    reads = store.reads
    new = resolved(recipe(name="new-name", max_context_length=32768))
    await store.put(new.name, new.model_dump(mode="json"))
    assert (await routes.select(new.name)).name == new.name
    assert store.reads == reads + 1


@pytest.mark.anyio
async def test_a_stale_cache_is_served_while_one_refresh_runs_in_the_background():
    base = resolved()
    routes, store = routes_for(base)
    await routes.definitions()
    added = resolved(recipe(name="added", max_context_length=32768))
    await store.put(added.name, added.model_dump(mode="json"))
    routes._loaded_at = time.monotonic() - 6
    reads = store.reads
    assert await names(routes) == [base.name]
    assert await names(routes) == [base.name]
    await routes._refresh
    assert store.reads == reads + 1
    assert set(await names(routes)) == {base.name, added.name}


@pytest.mark.anyio
async def test_a_failed_refresh_keeps_the_last_recipe_list():
    base = resolved()
    routes, store = routes_for(base)
    await routes.definitions()

    async def unavailable(prefix):
        raise ConnectionError("volume unavailable")

    store.list_keys = unavailable
    routes._loaded_at = time.monotonic() - 6
    assert await names(routes) == [base.name]
    await routes._refresh
    assert await names(routes) == [base.name]


@pytest.mark.anyio
async def test_unreadable_registry_files_are_skipped_and_kept():
    base = resolved()
    routes, store = routes_for(base)
    await store.put("invalid", {"recipe": {"name": "invalid"}})
    await store.put("not-json", {})
    read = store.get

    async def get(key):
        if key == "not-json":
            raise json.JSONDecodeError("truncated", "{", 1)
        return await read(key)

    store.get = get
    assert await names(routes) == [base.name]
    assert await read("invalid") is not None
    assert await read("not-json") is not None


@pytest.mark.anyio
async def test_the_registry_ignores_fields_a_newer_spindle_wrote():
    base = resolved()
    routes, store = routes_for(base)
    row = {**base.model_dump(mode="json"), "added_in_a_later_release": True}
    await store.put(base.name, row)
    with pytest.raises(ValueError):
        DeploymentConfig.model_validate(row)
    assert await names(routes) == [base.name]


def client_for(routes, store=None, **hooks):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_control_plane_app(
                ControlPlane(store or InMemoryKeyValueStore(), SimpleNamespace()),
                routes,
                api_key="test",
                **hooks,
            )
        ),
        base_url="http://frontend",
        headers={"x-api-key": "test"},
    )


@pytest.mark.anyio
async def test_every_recipe_routes_by_name_and_plain_models_pick_the_longest():
    kv = InMemoryKeyValueStore()
    base = resolved()
    longer = resolved(recipe(name="longer", max_context_length=65536))
    routes, _ = routes_for(base, longer)
    assert (await routes.select(base.model)).definition_id == "longer"
    assert (await routes.select(base.name)).definition_id == base.definition_id
    assert await routes.capabilities() == [
        {"model_name": base.model, "max_context_length": 65536},
        {"model_name": "longer", "max_context_length": 65536},
        {"model_name": base.name, "max_context_length": base.max_context_length},
    ]
    async with client_for(routes, kv) as client:
        session_id = (await client.post("/api/v1/create_session", json={})).json()[
            "session_id"
        ]
        for seq, model_name, expected in (
            (0, base.model, "longer"),
            (1, base.name, base.definition_id),
        ):
            response = await client.post(
                "/api/v1/create_model",
                json={
                    "session_id": session_id,
                    "model_seq_id": seq,
                    "base_model": model_name,
                    "lora_config": {"rank": 32},
                },
            )
            assert response.status_code == 200, response.text
            row = await kv.get(model_key(response.json()["model_id"]))
            assert row["engine_definition_id"] == expected


@pytest.mark.anyio
async def test_queues_report_each_recipe():
    async def list_instances():
        return ()

    plane = ControlPlane(
        InMemoryKeyValueStore(), SimpleNamespace(list_instances=list_instances)
    )
    spec = recipe(trainer_max_instances=3, trainer_max_clients_per_instance=4)
    app = create_control_plane_app(plane, [resolved(spec)], api_key="test")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://frontend",
        headers={"x-api-key": "test"},
    ) as client:
        response = await client.get("/api/v1/spindle/queues")
        assert response.status_code == 200, response.text
        [queue] = response.json()["queues"]
        assert queue["name"] == spec.name
        assert queue["backlog"] == {"count": 0, "oldest_age_s": None}
        assert queue["slots"] == {"used": 0, "free": 0, "per_trainer": 4}
        assert queue["trainers"] == {"running": 0, "starting": 0, "max": 3}
        assert queue["recent"] == {
            "window_s": 3600.0,
            "added": 0,
            "placed": 0,
            "released": 0,
            "wait_s": {"p50": None, "p95": None, "max": None},
        }


def test_native_boolean_opposite_flags_and_optional_value():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", dest="bias", action="store_true")
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.add_argument("--optional", nargs="?")
    parser.add_argument("--keep", action="store_true")
    argv = ["--bias", "--optional", "--keep"]
    apply_config_overrides(parser, {"bias": False, "optional": "supplied"}, argv)
    assert vars(parser.parse_args(argv)) == {
        "bias": False,
        "optional": "supplied",
        "keep": True,
    }
    parser.add_argument("--custom", action="append")
    with pytest.raises(ValueError, match="unsupported argparse action"):
        apply_config_overrides(parser, {"custom": [1]}, [])


def test_native_type_callbacks_receive_text():
    def readable_int(value):
        return int(value.strip().removesuffix("k")) * (
            1000 if value.endswith("k") else 1
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=readable_int)
    parser.add_argument("--sizes", nargs="+", type=readable_int)
    argv = []
    apply_config_overrides(parser, {"context_length": 65536, "sizes": [32, "2k"]}, argv)
    args = parser.parse_args(argv)
    assert args.context_length == 65536
    assert args.sizes == [32, 2000]


def test_native_sections_survive_serialization_without_allowlist():
    spec = recipe("qwen35-4b-fft-64k")
    data = vars(spec)
    data["megatron_cfg"]["provider_overrides"]["future_provider_option"] = {
        "layers": [1, 4],
        "enabled": False,
    }
    data["megatron_cfg"]["optimizer_overrides"] = {"future_optimizer_option": 0.125}
    data["megatron_cfg"]["distributed_overrides"] = {"future_ddp_option": False}
    spec = BaseConfig(**data)
    settings = backend_config(spec, "/assets/pinned")
    config, _ = parse_backend_config(json.loads(json.dumps(settings)))
    assert config.hf_checkpoint == "/assets/pinned"
    assert config.seq_length == spec.max_context_length
    assert config.provider_overrides["future_provider_option"] == {
        "layers": [1, 4],
        "enabled": False,
    }
    assert config.optimizer_overrides == {"future_optimizer_option": 0.125}
    assert config.distributed_overrides == {"future_ddp_option": False}
    assert config.optimizer.lr == 0.0001
    assert vars(spec) == data  # Building does not consume or mutate the config.


@pytest.mark.parametrize(
    "section,options,match",
    [
        ("provider_overrides", {"context_parallel_size": 4}, "managed"),
        ("optimizer_overrides", {"bf16": False}, "managed"),
        ("distributed_overrides", {"use_distributed_optimizer": False}, "managed"),
        ("provider_overrides", [], "mapping"),
        ("optimizer", {"optimizer": "sgd"}, "Adam"),
    ],
)
def test_megatron_cli_options_preserve_integration_contract(section, options, match):
    spec = recipe("qwen35-4b-fft-64k", **{f"megatron_cfg__{section}": options})
    with pytest.raises(ValueError, match=match):
        backend_config(spec)


def test_backend_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        backend_config(recipe(backend="missing"))


def test_new_miles_and_sglang_options_need_no_deployment_schema_change():
    spec = recipe(
        miles_cfg__cli_options__future_miles_option=[1, 2],
        sglang_cfg__future_sglang_option=False,
    )
    assert backend_config(spec)["miles"]["cli_options"]["future_miles_option"] == [
        1,
        2,
    ]
    assert serving_options(spec)["future_sglang_option"] is False


def test_fft_capacity_is_checked_by_backend_setup():
    with pytest.raises(ValueError, match="FFT trainers admit one client"):
        backend_config(recipe("qwen35-4b-fft-64k", trainer_max_clients_per_instance=2))


def test_reserved_environment_is_checked_by_modal_setup():
    with pytest.raises(ValueError, match="managed"):
        deployment_env({"SPINDLE_BACKEND_CONFIG": "oops"})
    assert deployment_env({"MY_SETTING": "value"}) == {"MY_SETTING": "value"}


def test_record_creation_copies_without_reparsing():
    spec = recipe()
    original = vars(spec)
    row = DeploymentConfig.create(spec)
    assert vars(spec) == original
    assert row.asset_path == f"/assets/{spec.model}"
    row.recipe.miles_cfg["max_lora_rank"] = 64
    assert spec.miles_cfg["max_lora_rank"] == 32
    saved = row.model_dump_json()
    assert DeploymentConfig.model_validate_json(saved).model_dump(
        mode="json"
    ) == row.model_dump(mode="json")


def test_python_config_composition(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(
        "from spindle.configs.qwen35_9b_lora_16k import Config as Parent\n"
        "class Config(Parent):\n"
        "    name = 'custom'\n"
        "    overrides = {'trainer_memory_mib': 123456}\n"
        "config = Config()\n"
    )
    custom = load(path)
    original = recipe()
    assert custom.name == "custom"
    assert custom.trainer_memory_mib == 123456
    assert custom.miles_cfg == original.miles_cfg
    assert original.trainer_memory_mib == 65536
    custom.trainer_max_instances = 9
    assert custom.trainer_max_instances == 9
    assert original.trainer_max_instances == 1


def test_worker_record_contains_resolved_settings(monkeypatch):
    record = resolved()
    assert record.trainer_settings["miles"]["actor_num_gpus_per_node"] == 4
    assert record.inference_settings["max_lora_rank"] == 32
    assert DeploymentConfig.model_validate_json(record.model_dump_json()).model_dump(
        mode="json"
    ) == record.model_dump(mode="json")


@pytest.mark.parametrize("source", ["Config = {}", "class Config: pass", "value = 1"])
def test_config_file_must_export_config_subclass(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(source)
    with pytest.raises(ValueError, match="BaseConfig instance"):
        load(path)


def test_config_import_error_preserves_traceback_and_restores_path(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("raise RuntimeError('bad user config')")
    before = list(sys.path)
    with pytest.raises(RuntimeError, match="bad user config"):
        load(path)
    assert sys.path == before


def test_no_yaml_config_ingestion(tmp_path):
    with pytest.raises(ValueError, match="Python .py"):
        load(tmp_path / "old.yaml")


def test_recipe_app_names_follow_the_recipe_name():
    base = resolved()
    assert base.trainer_app_name == f"spindle-trainer-{base.name}"
    assert base.inference_app_name == f"spindle-inference-{base.name}"
    named = resolved(recipe(platform__frontend="my-spindle"))
    assert named.trainer_app_name == f"my-spindle-trainer-{base.name}"
    assert named.inference_app_name == f"my-spindle-inference-{base.name}"

    inference = resolved(recipe(sglang_cfg__max_running_requests=24))
    assert inference.trainer_app_name == base.trainer_app_name
    assert inference.trainer_identity() == base.trainer_identity()
    assert inference.inference_identity() != base.inference_identity()

    trainer = resolved(recipe(miles_cfg__max_tokens_per_gpu=8192))
    assert trainer.trainer_app_name == base.trainer_app_name
    assert trainer.trainer_identity() != base.trainer_identity()
    assert trainer.inference_identity() == base.inference_identity()

    other = resolved(recipe("qwen35-9b-lora-64k"))
    assert other.trainer_app_name != base.trainer_app_name
    assert other.inference_app_name != base.inference_app_name
    assert "implementation" not in other.model_dump()


def test_examples_only_contain_model_infrastructure():
    for path in Path(config_path("qwen35-9b-lora-16k")).parent.glob("qwen*.py"):
        config = load(path)
        assert not hasattr(config, "deployment")
        assert not hasattr(config, "revision")


@pytest.mark.parametrize("value", ["invalid", 7])
def test_backend_parser_validates_configured_types_and_choices(value):
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=[1, 2], required=True)
    argv = ["--count", "1"]
    apply_config_overrides(parser, {"count": value}, argv)
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize(
    "section,field",
    [
        ("optimizer_overrides", "lr"),
        ("optimizer_overrides", "adam_eps"),
        ("provider_overrides", "calculate_per_token_loss"),
        ("provider_overrides", "attention_backend"),
        ("distributed_overrides", "overlap_grad_reduce"),
    ],
)
def test_managed_backend_values_fail_before_record_creation(section, field):
    base = recipe("qwen35-4b-fft-64k")
    candidate = deepcopy(base)
    candidate.megatron_cfg[section] = {field: 1}
    with pytest.raises(ValueError, match=field):
        DeploymentConfig.create(candidate)


def test_multinode_ownership_and_topology():
    config = load(config_path("qwen38-27b-lora-256k"))
    row = DeploymentConfig.create(config)
    miles = row.trainer_settings["miles"]
    assert miles["actor_num_nodes"] == 2
    assert miles["actor_num_gpus_per_node"] == 8
    assert miles["tensor_model_parallel_size"] == 2
    assert miles["context_parallel_size"] == 8
    invalid = deepcopy(config)
    invalid.miles_cfg["actor_num_nodes"] = 3
    with pytest.raises(ValueError, match="actor_num_nodes"):
        DeploymentConfig.create(invalid)


def test_config_inheritance_and_constructor_overrides_copy_nested_options():
    class Child(Parent):
        name = "child"
        max_context_length = 8192
        overrides = {"miles_cfg.max_tokens_per_gpu": 8192}

    first, second = Child(), Child(name="second")
    first.miles_cfg["target_modules"].append("extra")
    first.miles_cfg["cli_options"]["recompute_num_layers"] = 2
    assert second.name == "second"
    assert second.max_context_length == 8192
    assert second.trainer_gpu == "H100"
    assert backend_config(second)["miles"]["max_tokens_per_gpu"] == 8192
    assert Parent().max_context_length == 16384
    for config in (second, Parent()):
        assert "extra" not in config.miles_cfg["target_modules"]
        assert config.miles_cfg["cli_options"]["recompute_num_layers"] == 1


def test_backend_dictionary_assignment_replaces_inherited_values():
    class Child(Parent):
        inference_gpu = "H100"
        sglang_cfg = {"max_running_requests": 4}

    config = Child()
    assert config.inference_gpu == "H100"
    assert config.inference_max_replicas == 8
    assert config.sglang_cfg == {"max_running_requests": 4}


def test_overrides_compose_across_generations_and_constructor():
    class Child(Parent):
        overrides = {
            "trainer_gpu": "H200",
            "trainer_env.FIRST": "1",
            "miles_cfg.max_tokens_per_gpu": 8192,
            "sglang_cfg.future_option.nested": [1, 2],
        }

    class Grandchild(Child):
        overrides = {
            "miles_cfg.max_tokens_per_gpu": 4096,
            "trainer_env.SECOND": "2",
        }

    config = Grandchild(
        name="custom",
        overrides={"miles_cfg.max_tokens_per_gpu": 2048},
    )
    assert config.name == "custom"
    assert config.trainer_gpu == "H200"
    assert config.trainer_env["FIRST"] == "1"
    assert config.trainer_env["SECOND"] == "2"
    assert config.miles_cfg["max_tokens_per_gpu"] == 2048
    assert Grandchild().miles_cfg["max_tokens_per_gpu"] == 4096
    assert Child().miles_cfg["max_tokens_per_gpu"] == 8192
    config.sglang_cfg["future_option"]["nested"].append(3)
    assert Child.overrides["sglang_cfg.future_option.nested"] == [1, 2]
    assert Grandchild().sglang_cfg["future_option"]["nested"] == [1, 2]
    assert "FIRST" not in Parent().trainer_env


def test_override_values_replace_dictionaries_and_lists():
    class Child(Parent):
        overrides = {"trainer_env": {"FIRST": "1"}}

    class Grandchild(Child):
        overrides = {
            "trainer_env": {},
            "miles_cfg.target_modules": ["q_proj"],
        }

    config = Grandchild()
    assert config.trainer_env == {}
    assert config.miles_cfg["target_modules"] == ["q_proj"]
    assert Child().trainer_env == {"FIRST": "1"}
    # Constructor fields replace the inherited section, then overrides apply.
    config = Grandchild(
        sglang_cfg={}, inference_gpu="H100", overrides={"inference_gpu": "H200"}
    )
    assert config.inference_gpu == "H200"
    assert config.sglang_cfg == {}


@pytest.mark.anyio
async def test_api_adds_take_a_base_or_a_recipe_under_frontend_settings():
    base = resolved()
    routes, store = routes_for(base)
    variant = await routes.build(
        DeploymentRequest(
            name="variant", base=base.name, overrides={"trainer_gpu": "H200"}
        )
    )
    assert variant.name == "variant" and variant.recipe.trainer_gpu == "H200"
    assert variant.model == base.model
    file_recipe = {**vars(recipe()), "platform": {"frontend": "elsewhere"}}
    from_file = await routes.build(
        DeploymentRequest(name="from-file", recipe=file_recipe)
    )
    assert from_file.recipe.platform == base.recipe.platform
    for name, overrides, kwargs, match in (
        ("x", {}, {"base": "missing"}, "unknown base"),
        ("x", {}, {"base": base.name, "recipe": {}}, "exactly one"),
        ("x", {"trainer_gpuu": "H200"}, {"base": base.name}, "trainer_gpuu"),
        ("x", {"platform.modal.region": "r"}, {"base": base.name}, "platform"),
        ("x", {"max_context_length": "big"}, {"base": base.name}, "integer"),
        ("x", {"sweep_interval_s": 1}, {"base": base.name}, "lifecycle"),
    ):
        with pytest.raises(ValueError, match=match):
            await routes.build(
                DeploymentRequest(name=name, overrides=overrides, **kwargs)
            )
    assert set(store._values) == {base.name}


@pytest.mark.parametrize(
    "body",
    [
        {"overrides": {"trainer_gpuu": "H200"}},
        {"overrides": {"platform": {}}},
        {"overrides": {"trainer_gpus_per_node": "2"}},
        {"overrides": {"trainer_gpu": 5}},
        {"overrides": {"trainer_disk_mib": 5}},
        {"base": "missing", "overrides": {}},
        {"base": None, "overrides": {}},
        {"name": "Bad Name", "overrides": {}},
        {"name": "a" * 50, "overrides": {}},
    ],
)
@pytest.mark.anyio
async def test_api_add_invalid_requests_fail_before_the_registry_write(body):
    base = resolved()
    routes, store = routes_for(base)
    request = {"name": "variant", "base": base.name, **body}
    if "base" in body and body["base"] is None:
        del request["base"]
    async with client_for(routes) as client:
        response = await client.post("/api/v1/spindle/deployments", json=request)
    assert response.status_code == 400, response.text
    assert "Traceback" not in response.json()["message"]
    assert set(store._values) == {base.name}


@pytest.mark.anyio
async def test_api_add_routes_at_once_conflicts_409_and_identical_adds_are_idempotent():
    base = resolved()
    routes, store = routes_for(base)
    async with client_for(routes) as client:

        async def post(name, overrides):
            return await client.post(
                "/api/v1/spindle/deployments",
                json={"name": name, "base": base.name, "overrides": overrides},
            )

        await routes.definitions()
        first = await post("variant", {"trainer_gpu": "H200"})
        assert first.status_code == 202, first.text
        # Listed straight away, without waiting for the cache to expire.
        assert "variant" in [d.name for d in await routes.definitions()]
        assert first.json()["deployment"]["name"] == "variant"
        assert first.json()["config"]["recipe"]["trainer_gpu"] == "H200"
        assert (await routes.get("variant")).recipe.trainer_gpu == "H200"
        identical = await post("variant", {"trainer_gpu": "H200"})
        conflict = await post("variant", {"trainer_gpu": "B200"})
        base_conflict = await post(base.name, {"trainer_gpu": "B200"})
        listed = (await client.get("/api/v1/spindle/deployments")).json()
    assert identical.status_code == 202
    assert conflict.status_code == base_conflict.status_code == 409
    assert conflict.json()["error"] == "conflict"
    assert "different settings" in conflict.json()["message"]
    assert sorted(d["name"] for d in listed["deployments"]) == [
        base.name,
        "variant",
    ]
    assert store._values["variant"]["recipe"]["trainer_gpu"] == "H200"
    with pytest.raises(DeploymentConflict):
        await routes.add(
            await routes.build(
                DeploymentRequest(
                    name="variant", base=base.name, overrides={"trainer_gpu": "B200"}
                )
            )
        )


@pytest.mark.anyio
async def test_api_remove_stops_routing_and_unknown_names_are_404():
    base = resolved()
    routes, store = routes_for(base)
    async with client_for(routes) as client:
        added = await client.post(
            "/api/v1/spindle/deployments",
            json={"name": "variant", "base": base.name, "overrides": {}},
        )
        assert added.status_code == 202, added.text
        assert await routes.get("variant") is not None
        removed = await client.delete("/api/v1/spindle/deployments/variant")
        assert removed.status_code == 200, removed.text
        assert removed.json()["deployment"]["name"] == "variant"
        listed = (await client.get("/api/v1/spindle/deployments")).json()
        assert [d["name"] for d in listed["deployments"]] == [base.name]
        assert "variant" not in store._values
        assert await routes.get("variant") is None
        missing = await client.delete("/api/v1/spindle/deployments/variant")
        assert missing.status_code == 404
        assert missing.json()["error"] == "not_found"


@pytest.mark.anyio
async def test_adding_a_config_starts_its_app_deploy_and_the_listing_tracks_it():
    base = resolved()
    routes, store = routes_for(base)
    deploying = set()

    async def deploy_apps(definition):
        deploying.add(definition.name)

    async def apps_status(definition):
        if definition.name == base.name:
            return {"apps": "deployed"}
        return {"apps": "deploying" if definition.name in deploying else "missing"}

    hooks = {"deploy_apps": deploy_apps, "apps_status": apps_status}
    async with client_for(routes, store, **hooks) as client:
        added = await client.post(
            "/api/v1/spindle/deployments",
            json={"name": "variant", "base": base.name, "overrides": {}},
        )
        assert added.status_code == 202, added.text
        assert added.json()["deployment"]["apps"] == "deploying"
        listed = (await client.get("/api/v1/spindle/deployments")).json()
    assert deploying == {"variant"}
    assert {d["name"]: d["apps"] for d in listed["deployments"]} == {
        base.name: "deployed",
        "variant": "deploying",
    }
    async with client_for(routes, store) as client:
        listed = (await client.get("/api/v1/spindle/deployments")).json()
    assert all("apps" not in d for d in listed["deployments"])


@pytest.mark.anyio
async def test_removing_a_config_models_still_train_on_needs_force_and_stops_its_apps():
    base = resolved()
    routes, _ = routes_for(base)
    plane = InMemoryKeyValueStore()
    stopped = []

    async def stop_apps(definition):
        stopped.append(definition.name)

    model = ModelRecord(
        model_id="session:train:0",
        session_id="session",
        model_seq_id=0,
        engine_definition_id="variant",
        spec={},
        created_at=1.0,
    )
    await plane.put(model_key(model.model_id), model.model_dump())
    await plane.put(placement_key(model.model_id), {"model_id": model.model_id})
    async with client_for(routes, plane, stop_apps=stop_apps) as client:
        added = await client.post(
            "/api/v1/spindle/deployments",
            json={"name": "variant", "base": base.name, "overrides": {}},
        )
        assert added.status_code == 202, added.text
        refused = await client.delete("/api/v1/spindle/deployments/variant")
        assert refused.status_code == 409
        assert refused.json()["models"] == [
            {"model_id": model.model_id, "session_id": "session", "created_at": 1.0}
        ]
        assert stopped == []
        assert await routes.get("variant") is not None
        forced = await client.delete("/api/v1/spindle/deployments/variant?force=true")
        assert forced.status_code == 200, forced.text
        # The orphaned model still names the config, but it is gone: 404, not 409.
        again = await client.delete("/api/v1/spindle/deployments/variant")
        assert again.status_code == 404
    assert stopped == ["variant"]
    assert await routes.get("variant") is None


@pytest.mark.anyio
async def test_re_adding_after_a_version_bump_is_no_conflict():
    base = resolved()
    routes, store = routes_for(base)
    request = DeploymentRequest(name="variant", base=base.name)
    first = await routes.add(await routes.build(request))
    await store.put("variant", {**first, "spindle_version": "0.0.1"})
    await routes.add(await routes.build(request))
    assert (await store.get("variant"))["spindle_version"] == "0.0.1"


@pytest.mark.anyio
async def test_a_concurrent_add_under_the_same_name_is_a_conflict():
    base = resolved()
    routes, store = routes_for(base)
    rival = await routes.build(
        DeploymentRequest(
            name="variant", base=base.name, overrides={"trainer_gpu": "B200"}
        )
    )
    put = store.put

    async def put_then_overwritten(key, value):
        await put(key, value)
        await put(key, rival.model_dump(mode="json"))

    store.put = put_then_overwritten
    with pytest.raises(DeploymentConflict):
        await routes.add(
            await routes.build(DeploymentRequest(name="variant", base=base.name))
        )


def test_optimizer_overlap_changes_only_trainer_identity():
    base = resolved()
    overlap = resolved(trainer_overlap_optimizer_step=True)
    assert overlap.trainer_identity() != base.trainer_identity()
    assert overlap.inference_identity() == base.inference_identity()
