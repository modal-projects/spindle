from spindle.configs.qwen35_4b_fft_64k import Config as Parent

# One trainer. The two sampler recipes below are separate inference deployments
# attached to that trainer's published model.
TRAINER = Parent(
    name="qwen35-4b-mixed",
    trainer_region=(),
    trainer_rdma=True,
)

GROUPS = (
    {"name": "h100-bf16", "gpu": "H100", "region": (), "quantization": None},
    {"name": "h200-fp8", "gpu": "H200", "region": None, "quantization": "fp8"},
)


def sampler_config(group):
    overrides = {"sglang_cfg.tp_size": 1}
    if group["quantization"]:
        overrides["sglang_cfg.quantization"] = group["quantization"]
    return Parent(
        name=f"{TRAINER.name}-{group['name']}",
        inference_gpu=group["gpu"],
        inference_region=group["region"],
        inference_min_replicas=1,
        inference_max_replicas=1,
        overrides=overrides,
    )
