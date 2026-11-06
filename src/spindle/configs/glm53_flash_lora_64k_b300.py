"""Experimental single-node rank-16 GLM Flash with sampler expert replay.

Use sample_with_replay(return_routed_experts=True), include its replay tensors
in every training datum, and request train_unembed=True for lm_head. Clients
own the loss and optimizer schedule: e.g. importance_sampling over four calls
of 32 sequences, followed by one optim_step. The 65,536-token GPU budget keeps
40–48K prompts to one sequence per DP rank, regardless of that request size.
"""

from spindle.configs.glm53_flash_lora_48k_b300 import Config as SingleNodeConfig


class Config(SingleNodeConfig):
    name = "glm53-flash-lora-64k-b300"
    # 65,536-token trajectories plus sampler completion bookkeeping.
    max_context_length = 65552
    trainer_max_clients_per_instance = 1
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.max_lora_rank": 16,
        "miles_cfg.max_lora_slots": 1,
        "miles_cfg.default_lora_alpha": 16,
        "miles_cfg.target_modules": [
            *SingleNodeConfig.miles_cfg["target_modules"],
            "lm_head",
        ],
        "miles_cfg.cli_options.use_rollout_routing_replay": True,
        "miles_cfg.cli_options.global_batch_size": 32,
        "miles_cfg.cli_options.micro_batch_size": 1,
        "sglang_cfg.enable_return_routed_experts": True,
        "sglang_cfg.disable_radix_cache": True,
        "sglang_cfg.max_running_requests": 4,
        "sglang_cfg.max_total_tokens": 262144,
        "sglang_cfg.cuda_graph_max_bs_decode": 4,
    }


config = Config()
