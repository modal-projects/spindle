# Changelog

Notable changes to Spindle, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [semantic versioning](https://semver.org/). Before 1.0, a minor release (0.x.0) can change config defaults or deployment behavior; a patch release (0.x.y) only fixes bugs. Each release is tagged `vX.Y.Z` and published as a GitHub Release.

## Unreleased

## 0.2.0 - 2026-10-18

The first versioned release. Deployments made before it report `0.1.0`.

### Added

- Spindle reports its version and source commit: `spindle --version`, `spindle_version` and `spindle_commit` tags on every Modal app `spindle deploy` creates, and the same fields in `/api/v1/get_server_capabilities`.
- `trainer_gpu` and `inference_gpu` accept a list of GPU types in preference order (#89). `qwen36-35b-a3b-lora-32k` falls back from H200 to B300 and B200.
- Support for Tinker SDK 0.25 (#3).
- Model recipes with multi-LoRA backend support (#35), and a Qwen3.5 9B 256K config (#86).

### Changed

- **GPU placement is no longer pinned to `us-west` by default** (#4). `platform.modal.region` now defaults to `None`; set it only to restrict placement.
- SGLang v0.5.21 and a newer Stitch (#30).
- gpt-oss presets: train `lm_head` LoRA to match Tinker SDK defaults (#6), serve LoRA from BF16 weights (#7), and retuned trainer and sampler parallelism and admission limits (#36, #40, #45, #49).
- Sampling: multi-turn conversations route to one sampler replica (#46), and warm sampler pools persist between training steps (#37).
- Faster cold starts: TileLang kernels persist in the shared kernel cache (#82), and the asset download is skipped once weights are on the volume (#83).
- Lower request and training overhead (#29, #31, #41, #42).

### Fixed

- `spindle deploy` from macOS and other onboarding issues (#23).
- Miles CLI overrides applied after all argument groups are registered (#5).
- Miles LoRA A initialization matches PEFT and Tinker (#10).
- Grouped-expert LoRA checkpoint loading in the pinned Miles version (#16).
- `execute_sample` is non-preemptible (#15).
