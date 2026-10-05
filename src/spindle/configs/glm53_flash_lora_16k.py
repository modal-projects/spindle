from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "glm53-flash-lora-16k"
    model = "zai-org/GLM-5.3-Flash"
    max_context_length = 16384
    trainer_image = "spindle.providers.modal.glm53_image:trainer_image"
    inference_image = "spindle.providers.modal.glm53_image:inference_image"
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_nodes = 4
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_clients_per_instance = 4
    trainer_env = {"TORCHINDUCTOR_COMPILE_THREADS": "1"}
    miles_cfg = {
        "model_type": "",
        "tensor_model_parallel_size": 8,
        "expert_model_parallel_size": 32,
        "expert_tensor_parallel_size": 1,
        "max_tokens_per_gpu": 16384,
        "max_lora_slots": 4,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        # Keep targets in the language model. kv_b_proj uses an absorbed weight
        # whose upstream implementation cannot apply a different delta per slot.
        "target_modules": [
            "model.language_model.layers.*.self_attn.q_proj",
            "model.language_model.layers.*.self_attn.k_proj",
            "model.language_model.layers.*.self_attn.v_proj",
            "model.language_model.layers.*.self_attn.o_proj",
            "model.language_model.layers.*.self_attn.q_a_proj",
            "model.language_model.layers.*.self_attn.q_b_proj",
            "model.language_model.layers.*.self_attn.kv_a_proj_with_mqa",
            "model.language_model.layers.*.mlp.gate_proj",
            "model.language_model.layers.*.mlp.up_proj",
            "model.language_model.layers.*.mlp.down_proj",
            "model.language_model.layers.*.mlp.shared_experts.gate_proj",
            "model.language_model.layers.*.mlp.shared_experts.up_proj",
            "model.language_model.layers.*.mlp.shared_experts.down_proj",
            "model.language_model.layers.*.mlp.experts.*.gate_proj",
            "model.language_model.layers.*.mlp.experts.*.up_proj",
            "model.language_model.layers.*.mlp.experts.*.down_proj",
        ],
        "cli_options": {
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
            "rotary_base": 800000,
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
            "layernorm_epsilon": 1e-5,
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
    }
    inference_gpu = "H200"
    inference_gpus_per_node = 8
    inference_cpu = 32
    inference_memory_mib = 524288
    inference_target_concurrency = 16
    inference_max_replicas = 1
    inference_startup_timeout_s = 3600
    sglang_cfg = {
        "tp_size": 8,
        "ep_size": 8,
        "quantization": "fp8",
        "attention_backend": "dsa",
        "dsa_prefill_backend": "tilelang",
        "dsa_decode_backend": "tilelang",
        "kv_cache_dtype": "bfloat16",
        "linear_attn_backend": "triton",
        "moe_runner_backend": "triton",
        "disable_shared_experts_fusion": True,
        "mem_fraction_static": 0.8,
        "max_running_requests": 32,
        "max_queued_requests": 32,
        "max_loaded_loras": 8,
        "max_loras_per_batch": 4,
        "chunked_prefill_size": 4096,
        "disable_prefill_cuda_graph": True,
        "language_only": True,
    }


config = Config()
