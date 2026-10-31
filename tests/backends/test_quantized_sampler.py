"""The sampler requantizes BF16 publications; the trainer stays BF16."""

import json

from spindle.backends.deployment import resolve_backend_settings
from spindle.configs.qwen3_4b_lora_32k import Config as LoraConfig
from spindle.configs.qwen35_4b_fft_64k import config as bf16
from spindle.configs.qwen35_4b_fft_fp8_64k import config as fp8


def test_fp8_full_parameter_sampler_requantizes_cpu_staged_weights():
    trainer, inference = resolve_backend_settings(fp8, "/assets/pending")
    assert inference["quantization"] == "fp8"
    assert inference["weight_update_staging"] == "cpu"
    assert inference["weight_version"] == "0"
    _, plain = resolve_backend_settings(bf16, "/assets/pending")
    assert "quantization" not in plain
    assert "quantization" not in json.dumps(trainer)


def test_nvfp4_online_reaches_the_sampler_unchanged():
    class Nvfp4(type(bf16)):
        overrides = {"sglang_cfg.quantization": "nvfp4_online"}

    _, inference = resolve_backend_settings(Nvfp4(), "/assets/pending")
    assert inference["quantization"] == "nvfp4_online"
    assert inference["weight_update_staging"] == "cpu"


class QuantizedLora(LoraConfig):
    overrides = {"sglang_cfg.quantization": "fp8"}


def test_lora_quantization_loads_with_the_base_and_skips_update_staging():
    _, inference = resolve_backend_settings(QuantizedLora(), "/assets/pending")
    assert inference["quantization"] == "fp8"
    assert inference["enable_lora"] is True
    assert "weight_update_staging" not in inference
