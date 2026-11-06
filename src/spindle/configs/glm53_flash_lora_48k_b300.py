"""Single-node B300 GLM Flash: TP2, EP8, and one 48K row per DP rank.

The 9.31K input tokens/s measurement used the previous Miles image, rank 32,
24 sequences, and masked cross-entropy. Revalidate throughput on this image;
that measurement does not cover rank 16, lm_head, or rollout routing replay.
"""

from spindle.configs.glm53_flash_lora_64k import Config as GLMConfig


class Config(GLMConfig):
    name = "glm53-flash-lora-48k-b300"
    max_context_length = 49152
    trainer_gpu = "B300"
    trainer_nodes = 1
    trainer_rdma = True
    inference_gpu = "B300"
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.expert_model_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 49152,
    }


config = Config()
