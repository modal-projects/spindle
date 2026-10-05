"""Reduced GLM architecture check: two slots, updates, restore and PEFT export.

Invoked by validate_glm53.py --train. Uses random weights; this is a correctness
check for the integration, not a test of the public checkpoint's quality.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_runtime.runtime import MilesRuntime


def main():
    torch.manual_seed(42)
    path = Path("/tmp/glm53-test-model")
    config = AutoConfig.from_pretrained("zai-org/GLM-5.3-Flash")
    config.quantization_config = None
    text = config.text_config
    text.num_hidden_layers = 4
    text.hidden_size = 256
    text.intermediate_size = 512
    text.moe_intermediate_size = 256
    text.n_routed_experts = 8
    text.num_experts_per_tok = 2
    text.num_attention_heads = text.num_key_value_heads = 8
    text.q_lora_rank = 64
    text.vocab_size = 1024
    text.pad_token_id = 0
    text.eos_token_id = [1]
    text.num_nextn_predict_layers = 0
    text.layer_types = ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    text.mlp_layer_types = ["dense"] * 3 + ["sparse"]
    text.indexer_types = ["full"] * 4
    text.index_n_heads = 8
    text.index_topk = 16
    text.linear_attn_config = dict(text.linear_attn_config)
    text.linear_attn_config.update(
        num_heads=8, kda_layers=[0, 1, 2], full_attn_layers=[3]
    )
    config.vision_config.depth = 1
    config.vision_config.hidden_size = 64
    config.vision_config.num_heads = 4
    config.vision_config.intermediate_size = 128
    config.vision_config.out_hidden_size = 256
    config.vision_config.projection_intermediate_size = 128
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(path)
    del model
    print("Saved reduced random GLM checkpoint", flush=True)

    settings = json.loads(sys.argv[1])
    settings.update(
        hf_checkpoint=str(path),
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        max_tokens_per_gpu=256,
        max_lora_slots=2,
        max_lora_rank=8,
        default_lora_alpha=8,
        extra_args=["--seq-length", "256"],
    )
    settings["cli_options"].update(
        num_layers=4,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        q_lora_rank=64,
        num_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        vocab_size=1024,
        moe_layer_freq="[0]*3+[1]",
        global_batch_size=2,
        micro_batch_size=1,
    )
    runtime = MilesRuntime(MilesBackendConfig(**settings))
    try:
        for slot in (0, 1):
            runtime.load_slot(slot, 8, 8)
        rows = tuple(
            (
                slot,
                {
                    "tokens": list(range(10 + slot, 42 + slot)),
                    "target_tokens": list(range(11 + slot, 43 + slot)),
                    "target_len": 32,
                    "weights": [1.0] * 32,
                },
            )
            for slot in (0, 1)
        )

        def forward():
            return runtime.forward_backward(
                rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=True
            )

        before = forward()
        for step in range(2):
            out = runtime.forward_backward(
                rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=False
            )
            assert all(
                torch.isfinite(torch.tensor(row["logprobs"])).all() for row in out
            )
            metrics = runtime.optim_step(
                {
                    slot: {
                        "learning_rate": 1e-4,
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    }
                    for slot in (0, 1)
                }
            )
            print("UPDATE", step, metrics, flush=True)
        after = forward()
        assert any(a["logprobs"] != b["logprobs"] for a, b in zip(before, after))
        runtime.save_slot(0, "/tmp/glm53-test-checkpoint")
        runtime.unload_slot(0)
        runtime.load_slot(0, 8, 8, checkpoint="/tmp/glm53-test-checkpoint")
        restored = forward()
        torch.testing.assert_close(
            torch.tensor(after[0]["logprobs"]),
            torch.tensor(restored[0]["logprobs"]),
            rtol=0,
            atol=1e-5,
        )
        runtime.export_slot_peft(
            slot=0,
            path="/tmp/glm53-test-adapter",
            rank=8,
            alpha=8,
            base_model=str(path),
            target_modules=tuple(settings["target_modules"]),
            lora_dropout=0,
        )
        assert Path("/tmp/glm53-test-adapter/adapter_model.safetensors").is_file()
        print(
            "PASS: two-slot forward/backward, two updates, checkpoint restore, PEFT export",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
