---
name: spindle
description: Drive Tinker-compatible training and sampling on Spindle (Modal).
when_to_use: Deploying, connecting, configuring engines or sampler capacity, publishing weights, checkpointing, or diagnosing startup/throughput on Spindle.
---

# Spindle

## Connect or deploy

- Connect: `tinker.ServiceClient(base_url=url, api_key=key)`.
- Deploy: `uv run spindle deploy <config-name>`; bare `spindle deploy` is interactive and exits without a terminal.

## Shared vs scoped

- Shared deployment: one API serves many LoRA/FFT jobs; each FFT model gets a dedicated trainer.
- Scoped: `with spindle.run(engine=engine) as (url, api_key)`, a single-tenant FFT server for the block. Python 3.12, one active model, other processes can connect while the owner lives.
- LoRA: `service.create_lora_training_client(base_model=..., rank=...)`. FFT: `spindle.create_full_training_client(service, engine.model)`.

## Rules

- Publish weights before sampling: `training.save_weights_and_get_sampling_client()`.
- Latest pool gives min-version (may serve newer). Fixed-version eval: `save_weights_for_sampler(name)` then `create_sampling_client(path)`.
- Scope exit saves no checkpoint; retrying the process creates a new scope, so restore from an explicit `save_state`.
- A trainer waiting on rollouts stays allocated and billed.

## References

| Topic | File |
| --- | --- |
| Credentials, deploy commands, configs, teardown | [deploy.md](references/deploy.md) |
| Engine recipes, pools, cold starts, cost | [engines.md](references/engines.md) |
| Publication, checkpoints, recovery | [weights.md](references/weights.md) |
| Multi-LoRA clients, batching, pricing | [multi-lora.md](references/multi-lora.md) |
