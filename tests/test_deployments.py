import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from spindle.backends.deployment import backend_config, serving_options
from spindle.backends.megatron_config import parse_backend_config
from spindle.backends.miles_arguments import apply_config_overrides
from spindle.configs.qwen35_9b_lora_16k import Config as Parent
from spindle.configuration import BaseConfig
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.control_plane.keys import model_key
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
    ],
)
def test_invalid_integrations_fail_when_building_backend_settings(changes, match):
    spec = recipe(**changes)
    with pytest.raises(ValueError, match=match):
        backend_config(spec)
        serving_options(spec)


def test_asset_paths_follow_the_model():
    a = resolved()
    assert a.asset_path == f"/assets/{a.recipe.model}"
    b = resolved(recipe(model="other/Qwen3.5-9B-Base"))
    assert a.asset_path != b.asset_path


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


def test_multiple_models_same_http_service_and_old_binding_survives_switch():
    async def run():
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

    asyncio.run(run())


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
