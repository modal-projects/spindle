# Deployment configuration validation

PR #55 was checked against `main` on 2026-09-28. These results cover the flat Python recipes and independently deployed trainer/inference apps.

## CPU and packaging

- **701 tests passed, 1 skipped**, using Python 3.12 and the CI version of PyTorch (2.10.0 CPU). The skipped test imports SGLang, which is installed in the GPU image rather than the CPU test environment.
- All **15 packaged presets** resolve backend settings and round-trip through saved JSON from the built wheel.
- Ruff import/error checks, formatting, the public PyPI lockfile check, and `git diff --check` passed. Changed Python files contain no function-local imports.

Regression coverage includes inherited overrides, independent mutable settings, backend constructor forwarding, separate trainer/frontend registries, startup-failure expiry, existing-pool updates, platform-setting changes, and retrying a partially failed deployment. Megatron provider replacement preserves pretrained-weight loading hooks. LoRA presets select explicit language-model targets and retain the corresponding Tinker training flags. FFT session pool names fit Modal's app-name limit.

## Live GPU checks

The isolated frontend is `spindle-pr55-validation-20260928` in `modal-labs/spindle-deploy`, region `us-west`. It uses uniquely named variants of the 9B LoRA/16K and 4B FFT/64K recipes. Each trainer requests 4 H100s. Both test inference pools use one H100 replica kept warm during validation. LoRA inference was changed from the preset’s H200 request after Modal reported insufficient H200 capacity in `us-west`; the production preset still requests H200.

| Check | Result |
| --- | --- |
| 9B LoRA client A | 3 complete training/publication/sampling steps passed |
| 9B LoRA client B, sharing client A’s trainer | 2 initial steps plus 1 after the inference update passed |
| 4B full-parameter training | 3 complete training/publication/sampling steps passed |
| Inference-only update | SGLang `max_running_requests` changed from 32 to 24; the running worker and frontend lookup both reported 24 |
| Trainer continuity | Client B retained the same trainer instance and boot ID through the inference updates |

Every step returned finite training and sampling log probabilities. The frontend lookup initially returned the prior configuration during the deployment transition; it was checked again, along with the worker’s actual settings, before releasing the continuation. All completed clients unloaded without reported cleanup errors. The isolated validation apps and pools were stopped afterward.

These checks use 512 training tokens and a 16-token generation cap. They exercise the configured processes and interfaces, not maximum-context capacity, convergence, autoscaling under load, or throughput. Other model presets and multi-node topology have CPU coverage only.
