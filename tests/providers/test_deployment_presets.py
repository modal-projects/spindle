import pytest

from spindle.backends.deployment import backend_config, serving_options
from spindle.deployments import DeploymentConfig, config_path, load
from spindle.providers.modal.deployment_configs import configs_from_env


@pytest.mark.parametrize(
    "path",
    sorted(config_path("qwen35-9b-lora-16k").parent.glob("[!_]*.py")),
    ids=lambda p: p.stem,
)
def test_all_packaged_recipes_validate_offline(path):
    spec = load(path)
    config = backend_config(spec)
    assert config[spec.backend]["hf_checkpoint"] == "/assets/pending"
    serving_options(spec)
    if spec.parameterization == "lora":
        resolved = DeploymentConfig.create(spec)
        # Legacy Megatron selectors also match vision layers in current Bridge.
        assert set(spec.miles_cfg["target_modules"]) == set(
            resolved.inference_settings["lora_target_modules"]
        )


@pytest.mark.parametrize("preset", ["qwen35-35b-a3b-fft-64k", "qwen36-35b-a3b-fft-64k"])
def test_moe_recipes_preserve_trainer_expert_parallelism(preset):
    config = backend_config(load(config_path(preset)))["megatron"]
    assert config["tensor_model_parallel_size"] == 4
    assert config["context_parallel_size"] == 2
    assert config["expert_model_parallel_size"] == 8
    assert config["provider_overrides"]["moe_token_dispatcher_type"] == "alltoall"


def test_moe_rollout_preserves_attention_data_parallelism():
    spec = load(config_path("qwen35-35b-a3b-fft-64k"))
    options = serving_options(spec)
    assert options["tp_size"] == options["dp_size"] == options["ep_size"] == 4
    assert options["enable_dp_attention"] is True
    definition = DeploymentConfig.create(spec)
    assert definition.recipe.inference_gpus_per_node == 4
    assert definition.rollout_tensor_parallel_size == 1


@pytest.mark.parametrize("context,cp", [(16384, 1), (65536, 2), (131072, 4)])
def test_qwen38_context_parallel_token_budget(context, cp):
    spec = load(config_path(f"qwen38-27b-lora-{context // 1024}k"))
    config = backend_config(spec)["miles"]
    assert config["context_parallel_size"] == cp
    assert config["max_tokens_per_gpu"] == context // cp
    assert config["actor_num_gpus_per_node"] == 8


@pytest.mark.parametrize("context,gpus,cp", [(32768, 4, 1), (131072, 8, 2)])
def test_gpt_oss_120b_uses_its_own_weights_and_mxfp4_layout(context, gpus, cp):
    spec = load(config_path(f"gpt-oss-120b-lora-{context // 1024}k"))
    deployment = DeploymentConfig.create(spec)
    trainer = deployment.trainer_settings["miles"]
    serving = deployment.inference_settings

    assert deployment.weights_repo == "openai/gpt-oss-120b"
    assert trainer["hf_checkpoint"] == "/assets/openai/gpt-oss-120b"
    assert deployment.max_context_length == serving["context_length"] == context
    assert spec.inference_gpu == "H200"
    assert spec.inference_gpus_per_node == serving["tp_size"] == 1
    assert serving["moe_runner_backend"] == "marlin"
    assert trainer["actor_num_gpus_per_node"] == gpus
    assert trainer["tensor_model_parallel_size"] == 4
    assert trainer["expert_model_parallel_size"] == gpus
    assert trainer["context_parallel_size"] == cp
    assert trainer["cli_options"]["num_layers"] == 36
    assert trainer["cli_options"]["num_experts"] == 128
    assert "balance_data" not in trainer["cli_options"]
    assert "log_probs_chunk_size" not in trainer["cli_options"]


@pytest.mark.parametrize(
    "preset,parent",
    [
        (
            "nemotron35-super-vl-120b-a12b-lora-64k",
            "nemotron3-super-120b-a12b-lora-64k",
        ),
        (
            "nemotron35-super-vl-120b-a12b-base-lora-64k",
            "nemotron3-super-120b-a12b-lora-64k",
        ),
        (
            "nemotron35-super-vl-120b-a12b-lora-64k-b300",
            "nemotron3-super-120b-a12b-lora-64k-b300",
        ),
    ],
)
def test_nemotron35_super_vl_trains_the_super_language_model(preset, parent):
    spec = load(config_path(preset))
    super_spec = load(config_path(parent))
    assert spec.model.startswith("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B")
    trainer = backend_config(spec)["miles"]
    assert trainer["model_type"] == "nemotron-3-super-120b-a12b"
    assert trainer == backend_config(super_spec)["miles"]
    assert serving_options(spec) == serving_options(super_spec)


def test_single_client_recipe_keeps_shared_backend_capacity():
    shared = load(config_path("qwen35-9b-lora-16k"))
    single = load(config_path("qwen35-9b-lora-16k-single"))
    assert single.trainer_max_clients_per_instance == 1
    assert (single.trainer_gpu, single.trainer_gpus_per_node, single.trainer_nodes) == (
        shared.trainer_gpu,
        shared.trainer_gpus_per_node,
        shared.trainer_nodes,
    )
    assert backend_config(single) == backend_config(shared)


@pytest.mark.parametrize("contents", [None, "[]", "{}"])
def test_missing_configs_have_no_python_catalog_fallback(
    monkeypatch, tmp_path, contents
):
    if contents is None:
        monkeypatch.delenv("SPINDLE_DEPLOYMENT_CONFIGS_FILE", raising=False)
    else:
        path = tmp_path / "configs.json"
        path.write_text(contents)
        monkeypatch.setenv("SPINDLE_DEPLOYMENT_CONFIGS_FILE", str(path))
    with pytest.raises(ValueError, match="configs"):
        configs_from_env()
