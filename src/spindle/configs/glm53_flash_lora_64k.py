"""Experimental 64K GLM-5.3 Flash LoRA with native cache-aware routing."""

from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "glm53-flash-lora-64k"
    model = "zai-org/GLM-5.3-Flash"
    model_weights = "zai-org/GLM-5.3-Flash-BF16"
    max_context_length = 65552
    trainer_image = "miles-v0.1.2"
    inference_image = "spindle.providers.modal.glm53_image:inference_image"
    trainer_gpu = "B200"
    trainer_gpus_per_node = 8
    trainer_nodes = 2
    trainer_cpu = 64
    trainer_memory_mib = 524288
    trainer_max_clients_per_instance = 4
    trainer_env = {"TORCHINDUCTOR_COMPILE_THREADS": "1"}
    miles_cfg = {
        "model_type": "",
        "tensor_model_parallel_size": 8,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        "max_tokens_per_gpu": 65552,
        "max_lora_slots": 4,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        # Suffix globs cover dense, shared, and routed expert projections.
        "target_modules": [
            "*.q_proj",
            "*.k_proj",
            "*.v_proj",
            "*.b_proj",
            "*.f_a_proj",
            "*.f_b_proj",
            "*.g_a_proj",
            "*.g_b_proj",
            "*.o_proj",
            "*.q_a_proj",
            "*.q_b_proj",
            "*.kv_a_proj_with_mqa",
            "*.gate_proj",
            "*.up_proj",
            "*.down_proj",
        ],
        "cli_options": {
            "use_rollout_routing_replay": False,
            "num_layers": 45,
            "hidden_size": 4096,
            "ffn_hidden_size": 12288,
            "num_attention_heads": 64,
            "multi_latent_attention": True,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_pos_emb_head_dim": 0,
            "v_head_dim": 256,
            "kv_channels": 256,
            "qk_layernorm": True,
            "rotary_base": 10000,
            "moe_layer_freq": "[0]*3+[1]*42",
            "num_experts": 288,
            "moe_router_topk": 8,
            "moe_router_score_function": "sigmoid",
            "moe_router_pre_softmax": True,
            "moe_router_enable_expert_bias": True,
            "moe_router_bias_update_rate": 0,
            "moe_router_topk_scaling_factor": 2.5,
            "moe_ffn_hidden_size": 2048,
            "moe_shared_expert_intermediate_size": 2048,
            "moe_router_dtype": "fp32",
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "vocab_size": 154880,
            "make_vocab_size_divisible_by": 16,
            "normalization": "RMSNorm",
            "layernorm_epsilon": 1e-05,
            "swiglu": True,
            "activation_func_clamp_value": 10,
            "add_bias_linear": False,
            "untie_embeddings_and_output_weights": True,
            "apply_rope_fusion": False,
            "enable_experimental": True,
            "dsa_attention_backend": "tilelang",
            "qkv_format": "thd",
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
        "context_parallel_size": 1,
    }
    inference_gpu = "B200"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 524288
    inference_target_concurrency = 64
    # Keep one replica through trainer updates; full-model cold starts take minutes.
    inference_min_replicas = 1
    inference_max_replicas = 4
    inference_startup_timeout_s = 3600
    inference_env = {"SGLANG_ENABLE_LORA_CSGMV_SPLIT_K": "1"}
    sglang_cfg = {
        "tp_size": 8,
        "ep_size": 8,
        "quantization": None,
        "attention_backend": "dsa",
        "dsa_prefill_backend": "tilelang",
        "dsa_decode_backend": "tilelang",
        "kv_cache_dtype": "bfloat16",
        "linear_attn_backend": "triton",
        "moe_runner_backend": "triton",
        "disable_shared_experts_fusion": True,
        "mem_fraction_static": 0.85,
        "max_running_requests": 64,
        "max_queued_requests": 256,
        "max_loaded_loras": 8,
        "max_loras_per_batch": 4,
        "chunked_prefill_size": 8192,
        "disable_prefill_cuda_graph": True,
        "enable_multimodal": False,
        "enable_return_routed_experts": False,
        "disable_radix_cache": False,
        "lora_backend": "csgmv",
        "enable_lora_overlap_loading": True,
        "cuda_graph_max_bs_decode": 64,
        "model_loader_extra_config": {
            "enable_multithread_load": True,
            "num_threads": 2,
        },
    }


config = Config()
