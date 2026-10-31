"""Flat Python recipes. Modal and the backend validate their own options."""

from copy import deepcopy


def _apply_overrides(values, overrides):
    for path, value in overrides.items():
        parts = path.split(".")
        target = values
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = deepcopy(value)


class BaseConfig:
    platform = {
        "frontend": "spindle",
        "modal": {"environment": None, "region": None},
        "secrets": {
            "api": "spindle-api",
            "sampler_proxy": "spindle-proxy",
            "huggingface": "huggingface-secret",
        },
        "storage": {
            "assets": "spindle-model-assets",
            "checkpoints": "spindle-checkpoints",
            "bulletin": "spindle-snapshot-bulletin",
        },
    }

    name = ""
    model = ""
    # Hugging Face repo for base weights; defaults to ``model``.
    model_weights = ""
    # Commit of the weights repo to download; empty follows its default branch.
    weights_revision = ""
    max_context_length = 16384
    parameterization = "lora"
    backend = "miles"

    trainer_gpu = "H100"
    # None uses platform.modal.region. () schedules in any region.
    trainer_region = None
    trainer_gpus_per_node = 1
    trainer_nodes = 1
    trainer_rdma = False
    trainer_cpu = 8
    trainer_memory_mib = 32768
    trainer_disk_mib = 0
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 1
    # Clients whose forward_backward requests may share one trainer call; None packs all.
    trainer_max_clients_per_batch = None
    trainer_timeout_s = 86400
    trainer_env = {}
    # A name from TRAINER_IMAGES in spindle.providers.modal.deployment_apps; empty uses the default.
    trainer_image = ""
    sampler_persistence_concurrency = 8

    inference_gpu = "H100"
    # None uses platform.modal.region. () schedules compute in any region.
    inference_region = None
    inference_gpus_per_node = 1
    inference_cpu = 8
    inference_memory_mib = 32768
    inference_disk_mib = 0
    inference_min_replicas = 0
    inference_max_replicas = 8
    inference_target_concurrency = 16
    # None measures the rate from the pool's own prefills.
    inference_routing_prefill_tokens_per_second = None
    # None allows a gap of 40% of max_running_requests, at least 4.
    inference_routing_balance_absolute = None
    inference_routing_balance_relative = 1.25
    inference_routing_cache_threshold = 0.5
    # None picks cache_wait for LoRA pools with one SGLang data-parallel rank,
    # and affinity otherwise.
    inference_lora_routing = None
    inference_scaledown_window_s = 300
    inference_startup_timeout_s = 1200
    inference_env = {}

    miles_cfg = {}
    megatron_cfg = {}
    sglang_cfg = {}
    session_idle_timeout_s = 300
    pool_idle_timeout_s = 300
    sweep_interval_s = 300

    def __init__(self, **kwargs):
        # Copy each ancestor's values before applying its overrides.
        values = {}
        for cls in reversed(type(self).__mro__):
            values.update(
                deepcopy(
                    {
                        key: value
                        for key, value in vars(cls).items()
                        if not key.startswith("_")
                        and key != "overrides"
                        and not callable(value)
                    }
                )
            )
            _apply_overrides(values, vars(cls).get("overrides", {}))
        overrides = kwargs.pop("overrides", {})
        values.update(deepcopy(kwargs))
        _apply_overrides(values, overrides)
        self.__dict__.update(values)
