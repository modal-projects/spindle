# Changelog

Notable changes to Spindle, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [semantic versioning](https://semver.org/). Before 1.0, a minor release (0.x.0) can change config defaults or deployment behavior; a patch release (0.x.y) only fixes bugs. Each release is tagged `vX.Y.Z` and published as a GitHub Release.

## Unreleased

### Added

- `deepseek_v41_flash_lora_8k` falls back to B200 and H200 for the trainer and samplers.
- DeepSeek-V4.1-Flash LoRA recipe with 32K context, `deepseek_v41_flash_lora_32k`. It splits each layer across eight GPUs instead of four.
- DeepSeek-V4.1-Flash-1017 LoRA recipes, `deepseek_v41_flash_1017_lora_8k` and `deepseek_v41_flash_1017_lora_32k`, which load the 1017 release.

### Changed

- DeepSeek-V4.1-Flash recipes load DeepSeek's FP8/FP4 release instead of a BF16 export. The trainer dequantizes it to BF16 as it loads, and samplers serve it with FP8 routed experts. Redeploying downloads the release once.

### Fixed

- DeepSeek-V4.1-Flash trainers on B200 no longer hang in their first step.
- Miles' gloo process group waits as long as `distributed_timeout_minutes`, so trainers on slow-loading hosts no longer time out after 30 minutes.
- The Miles v0.1.2 trainer image and the SGLang image are pinned to fork commits, so a cached build cannot serve an older fork branch.

## 0.1.1 - 2026-10-23

The first versioned release. Deployments made before it report `0.1.0`.

### Added

- Spindle reports its version and source commit: `spindle --version`, `spindle_version` and `spindle_commit` (short SHA) tags on every Modal app Spindle creates, and the version and full SHA in `/api/v1/get_server_capabilities`.
- Spindle apps carry `_modal_source`, `spindle_role`, job type, framework and model tags for usage reporting (#99).
- `trainer_gpu` and `inference_gpu` accept a list of GPU types in preference order (#89). `qwen36-35b-a3b-lora-32k` falls back from H200 to B300 and B200.
- `spindle deploy` interactive mode (#91) and a check for required environment variables (#92).
- Support for Tinker SDK 0.25 through 0.33 (#3, #69).
- Multiple target tokens per input position (#104).
- Model recipes with multi-LoRA backend support (#35), a Qwen3.5 9B 256K config (#86), and output-head LoRA for Qwen3.6 35B (#103).
- DeepSeek-V4.1-Flash LoRA recipe, `deepseek_v41_flash_lora_8k`, trained and served on B300s from pinned BF16 weights (#73, #74, #75).
- `trainer_image` selects a named trainer image, and multi-LoRA Miles recipes can run on Miles v0.1.2. LoRA target modules accept wildcard checkpoint paths (#74).
- While a trainer waits for GPU capacity, `create_model` reports which GPUs it is waiting for and for how long, instead of "starting a trainer" (#113).
- Per-phase startup timing on trainer and sampler containers (#58).

### Changed

- **GPU placement is no longer pinned to `us-west` by default** (#4). `platform.modal.region` now defaults to `None`; set it only to restrict placement.
- SGLang v0.5.21 and a newer Stitch (#30).
- gpt-oss presets: train `lm_head` LoRA to match Tinker SDK defaults (#6), serve LoRA from BF16 weights (#7), and retuned trainer and sampler parallelism and admission limits (#36, #40, #45, #49).
- Sampling: multi-turn conversations route to one sampler replica (#46), and warm sampler pools persist between training steps (#37).
- Faster cold starts: TileLang kernels persist in the shared kernel cache (#82), the asset download is skipped once weights are on the volume (#83), and `spindle deploy` builds the sampler image up front (#95).
- `spindle deploy` starts each model's base-weight download in the background. `create_model` and `create_sampling_client` return immediately, report download progress while they wait, and work for checkpoints that take longer than a request timeout to download (#107).
- Deployment configs reach the frontend as a file instead of an environment variable (#94).
- Optimizers other than AdamW are rejected instead of silently running AdamW (#78, #101).
- A client that asks to train the unembedding, as Tinker clients do by default, keeps it frozen when the deployment doesn't target it, instead of being refused (#74).
- Lower request and training overhead (#29, #31, #41, #42).

### Fixed

- `spindle deploy` from macOS and other onboarding issues (#23).
- Miles CLI overrides applied after all argument groups are registered (#5).
- Miles LoRA A initialization matches PEFT and Tinker (#10).
- Grouped-expert LoRA checkpoint loading in the pinned Miles version (#16).
- `execute_sample` is non-preemptible (#15).
- `load_state` on multi-node LoRA trainers, which failed on missing checkpoint shards (#74).
