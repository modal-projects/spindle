from mixed_samplers import config as common
from mixed_samplers import config_lora

from spindle.configs.qwen36_35b_a3b_lora_32k import Config as QwenConfig


def test_trainer_resources_and_parallelism():
    cfg = config_lora.config
    assert cfg.name == "qwen36-mimo-mixed-lora-256k"
    assert cfg.model_revision == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert cfg.max_context_length == 262144
    assert (cfg.backend, cfg.parameterization) == ("miles", "lora")
    assert (cfg.trainer_gpu, cfg.trainer_gpus_per_node, cfg.trainer_region) == (
        "B200",
        8,
        (),
    )
    assert (cfg.trainer_cpu, cfg.trainer_memory_mib) == (32, 786432)
    assert {
        key: cfg.miles_cfg[key]
        for key in (
            "tensor_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        )
    } == {
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 4,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    }
    assert cfg.miles_cfg["cli_options"] == {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
    }
    assert cfg.miles_cfg["max_tokens_per_gpu"] == 65536
    assert cfg.miles_cfg["max_lora_rank"] == 32
    assert cfg.miles_cfg["default_lora_alpha"] == 32
    assert cfg.miles_cfg["max_lora_slots"] == cfg.trainer_max_clients_per_instance == 6
    assert config_lora.LEARNING_RATE == 1e-5


def test_only_attention_and_vl_shared_expert_are_adapted():
    targets = config_lora.config.miles_cfg["target_modules"]
    assert targets == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        *[
            f"model.language_model.layers.*.mlp.shared_expert.{name}"
            for name in ("gate_proj", "up_proj", "down_proj")
        ],
    ]
    assert all(".mlp.experts." not in target for target in targets)
    assert QwenConfig.miles_cfg["target_modules"][-3:] == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_shared_heterogeneous_sampler_fleet_and_nvfp4_backend():
    assert len(config_lora.GROUPS) == 8
    for group, original in zip(config_lora.GROUPS, common.GROUPS, strict=True):
        assert {k: group[k] for k in ("name", "gpu", "gpus", "quantization")} == {
            k: original[k] for k in ("name", "gpu", "gpus", "quantization")
        }
        cfg = config_lora.sampler_config(group)
        assert cfg.sglang_cfg["quantization"] == original["quantization"]
        assert cfg.sglang_cfg["sampling_backend"] == "pytorch"
        assert cfg.sglang_cfg["sampling_mask_max_tokens"] == 248320
        assert cfg.sglang_cfg["kv_cache_dtype"] == "fp8_e4m3"
        assert (
            cfg.sglang_cfg["mem_fraction_static"]
            == QwenConfig.sglang_cfg["mem_fraction_static"]
        )
        assert (cfg.inference_cpu, cfg.inference_memory_mib) == (16, 262144)
        assert (
            cfg.inference_min_replicas,
            cfg.inference_max_replicas,
            cfg.inference_target_concurrency,
            cfg.inference_startup_timeout_s,
        ) == (1, 2, 16, 2400)
        assert cfg.inference_gpus_per_node == group["gpus"]
        assert cfg.sglang_cfg["tp_size"] == cfg.sglang_cfg["ep_size"] == group["gpus"]
        if group["quantization"] == "nvfp4_online":
            assert cfg.sglang_cfg["moe_runner_backend"] == "experimental_sgl_trtllm"
            assert original["moe_runner_backend"] == "flashinfer_trtllm"


def test_platform_resources_are_separate():
    platform = config_lora.config.platform
    assert platform["frontend"] == common.config.platform["frontend"] + "-lora"
    assert platform["secrets"]["api"] == "spindle-mimo-mixed-api-lora"
    for key, value in platform["storage"].items():
        assert value == common.config.platform["storage"][key] + "-lora"
