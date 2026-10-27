# Engines, pools, cold starts

## Recipes

`spindle.engines` returns frozen dataclasses; customize with `dataclasses.replace`, including nested:

```python
engine = replace(
    original,
    name="my-recipe",
    sampling=replace(original.sampling, memory_fraction=0.90),
    training=replace(original.training, seed=42),
)
```

Bundled: `qwen3_5_4b_full_64k()` (4×H100 trainer, 1×H100 sampler), `qwen3_5_9b_full_64k()` (8×H200 TP2×CP2×DP2), `qwen3_6_27b_full_64k()` (8×H200 TP4, 4×H200 sampler).

`Engine` fields: `name`, `model`, `trainer_gpu`/`sampler_gpu` (`"H100:4"` = type:count), `training` (`EngineModelConfig`), `sampling` (`SamplingConfig`), `revision`, `trainer_image`/`sampler_image` (`modal.Image`), `trainer_*`/`sampler_*` cpu+memory, `trainer_timeout` (1–86400 s), `backend_env` (`SPINDLE_*` keys reserved). Exports: `Engine`, `EngineModelConfig`, `OptimizerConfig`, `SamplingConfig`.

Changing GPU type: adjust count, parallelism, and context to fit model memory. `validate()` requires sampler parallelism to divide sampler GPU count.

## Sampler capacity

Scoped:

```python
with spindle.run(engine=engine, warm=True,
                 latest=spindle.Pool(min_containers=1, max_containers=8, scaledown_window=600)) as (url, key):
```

`Pool` defaults 0/8/300. `warm=True` waits for trainer load before the `with` body. Pool min activates at model creation; base and pinned samplers stay at 0.

Shared deployment (FFT clients only, at creation):

```python
create_full_training_client(service, model,
    rollout={"min_containers": 2, "max_containers": 8, "scaledown_window": 600})
```

Scoped clients reject `rollout`. `user_metadata` never configures resources; it carries only `run_id`/`attempt_id` telemetry labels.

Sizing: `min_containers` = capacity the job keeps busy continuously; `max_containers` = concurrency/budget; raise `scaledown_window` when sync batches would repeatedly cold start.

## Cold starts

Sampling requests get HTTP 408 until SGLang replicas are ready. Measure startup separately from steady state; samplers also go cold on scaledown or a new pinned pool.

| Operation | Startup work |
| --- | --- |
| Enter `spindle.run(warm=True)` | Prepare assets, allocate trainer, load model (`warm=False` defers) |
| Create full training client | Trainer readiness incl. load + distributed init if cold |
| First forward/backward | Kernel compile |
| First sample on cold pool | Allocate inference GPUs, load + compile model, apply published weights |

## Cost

GPU $ = allocated GPUs × seconds, trainer and inference as separate allocations; a small batch still pays the engine's full GPU layout. Use Modal container logs/GPU metrics plus optional OTLP traces for latency.
