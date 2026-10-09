from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "gpt-oss-20b-lora-64k"
    model = "openai/gpt-oss-20b"
    # SGLang LoRA MoE needs unquantized experts; this is the MXFP4 release upcast to BF16.
    model_weights = "lmsys/gpt-oss-20b-bf16"
    max_context_length = 65536
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_max_clients_per_instance = 8
    # The bf16 model fits on one 141 GB GPU, so each GPU holds a full replica (TP1/EP1, 8-way
    # data parallel) and skips tensor- and expert-parallel communication. At TP1 a single
    # sequence can be up to about 52k tokens on a 141 GB GPU; the vocabulary-sized logits of
    # longer sequences need tensor parallelism, which shards them. TP1 does not fit on 80 GB
    # GPUs such as H100; there, use tensor_model_parallel_size=4,
    # expert_model_parallel_size=8 and selective recompute (recompute_granularity="selective",
    # recompute_modules=["core_attn", "moe_act", "layernorm"]).
    miles_cfg = {
        "model_type": "gpt-oss-20b",
        "tensor_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
            "lm_head",
        ],
        "max_tokens_per_gpu": 32768,
        "max_lora_slots": 8,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            # Packed sink attention requires cuDNN >= 9.18 (Miles v0.1.0 has 9.22).
            "qkv_format": "thd",
            "attention_backend": "fused",
            "moe_permute_fusion": False,
            # Bounds the fp32 [tokens, vocab] logits buffer of the log-prob pass at TP1.
            "log_probs_chunk_size": 4096,
            "recompute_granularity": "selective",
            "recompute_modules": ["core_attn", "moe_act", "layernorm"],
        },
    }
    trainer_env = {
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        # Expert-LoRA checkpoint-load workaround; see miles_runtime/expert_lora_compat.py.
        "EXPERT_LORA_COMPAT": "1",
    }
    inference_gpu = "H200"
    inference_gpus_per_node = 4
    sglang_cfg = {
        "tp_size": 4,
        "dtype": "bfloat16",
        "mem_fraction_static": 0.7,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "max_loaded_loras": 64,
        "max_loras_per_batch": 8,
        "schedule_policy": "lpm",
        "moe_runner_backend": "triton",
    }


config = Config()
