---
name: spindle-dev
description: Contributor guide for modifying the Spindle repo itself.
when_to_use: Changing Spindle internals (configs, backends, control plane, engine, publication, sampling), upstreaming a config fix, or writing a validation entry. Not for running training.
---

# Developing Spindle

## Architecture

Three layers (docs/design.md):

- Control plane: stateless autoscaling web server (sessions, model placement, futures, sampling orchestration); state in Modal Dicts. `control_plane/service.py`, `providers/modal/kv.py`, `providers/modal/trainer_reconciler.py`.
- Training engine: model state + distributed backend. `engine/api.py` protocol, `engine/server.py` scheduling + batching, `engine/spmd.py` rank broadcast, `engine/backend_http.py` subprocess bridge.
- Sampling: published weights through Stitch bulletin volumes. `inference/bulletin.py` (LoRA adapters), `inference/fft_bulletin.py` (FFT deltas), `providers/modal/*_pool.py`.

Lanes: GPU ops serialize; `capture_*` runs on the GPU lane, `persist_*`/`publish_*` async off it. Backends must be lock-free: `capture_*` returns detached CPU state the persist call exclusively owns.

## Add a model config

1. `src/spindle/configs/<name>.py`: `Config(BaseConfig)` or subclass a bundled config; export `config = Config()`. Flat attributes + backend dicts (`miles_cfg`/`megatron_cfg`/`sglang_cfg`) + dotted `overrides`. No registry.
2. Add its path to `scripts/deploy_models.sh` and run it.
- Limits: `trainer.max_instances`, `inference.min_replicas`/`max_replicas`. Platform settings must match across the config set.
- Verify: `uv run --no-sync pytest -q` (preset resolution + JSON round-trip) and `spindle config validate`; live-check on an isolated frontend per docs/deployment-validation.md.

## Task references

- Add a backend: [add-backend.md](references/add-backend.md)
- Improve a bundled config and upstream the fix: [upstream-config.md](references/upstream-config.md)

## Validation entries

- CI (tests.yml): `uv sync --locked --python 3.12`, `pre-commit --all-files`, CPU torch 2.10.0, `uv run --no-sync pytest -q`, `uv build`; SDK matrix min/pinned/latest. Codegolf workflow: pytest, ruff, `codegolf config`, `render.py`.
- Entry shape (docs/validation.md, lora_validation.md, scoped-e2e-results.md, deployment-validation.md): environment, PR head, hardware, workload, results, coverage limits, versions, evidence paths; confirm all apps stopped with zero containers.
- Generators: `e2e_longrlvr_*_{full,miles}.py`, `e2e_terminal_bench_*`, `run_dapo_math_spindle_full.py`, `check_multi_targets_gpu.py`, `examples/codeforces-codegolf`.
