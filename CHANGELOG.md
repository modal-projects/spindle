# Changelog

Notable changes to Spindle, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [semantic versioning](https://semver.org/). Before 1.0, a minor release (0.x.0) can change config defaults or deployment behavior; a patch release (0.x.y) only fixes bugs. Each release is tagged `vX.Y.Z` and published as a GitHub Release.

## Unreleased

### Added

- Qwen3.8-27B LoRA recipe with B200 samplers, `qwen38_27b_lora_16k_b200`, which serves with the TensorRT-LLM attention backend and 64-token pages (#145).
- LoRA samplers can route each request to the replica with its longest cached prefix with `inference_lora_routing = "cache_wait"`, while keeping request load and pinned KV memory within bounds. The router runs in Rust inside each sampling container. It measures the pool's prefill rate from completed requests unless `inference_routing_prefill_tokens_per_second` sets it, and `inference_routing_balance_absolute`, `inference_routing_balance_relative` and `inference_routing_cache_threshold` tune it. Pools without routable replicas, including pools scaled to zero and pools whose SGLang cache publishes no observations, fall back to Modal's gateway routing (#121, #122, #123).

### Changed

- LoRA recipes use `cache_wait` routing by default when SGLang runs one data-parallel rank, and Modal session affinity otherwise. `inference_lora_routing = "affinity"` keeps session affinity. `inference_routing_balance_absolute` defaults to 40% of `max_running_requests`, and at least 4 (#123).
- LoRA samplers enable SGLang's CSGMV split-K kernel. For rank-32 adapters on models with 4,096-wide hidden layers, such as Qwen3.5-9B and Qwen3-8B, it speeds up the LoRA kernels for decode-sized batches by up to 2.9× on H200 and 2.6× on B200. Set `SGLANG_ENABLE_LORA_CSGMV_SPLIT_K=0` in `inference_env` to disable it (#123).

### Changed

- The frontend server runs in `platform.modal.region`, or `us-east` when no region is set. Only the CPU frontend is placed; trainers and samplers stay unpinned unless the recipe sets a region. Before, frontend containers could start in any region, and one on another continent added 2 s or more to each training step.
- `forward` requests use the same binary transport and protobuf results as `forward_backward`, and queued forwards are batched into one trainer call.
- The Miles backend prepares labels and weights with numpy instead of per-value Python loops. For 128 examples with `[seq, 5]` labels this saves about 1 s per training call.

### Changed

- When every datum in a batch has the same number of candidate labels (`target_tokens` shaped `[sequence_length, K]`), the Miles backend scores all K labels from one softmax over the vocabulary instead of K (#157).

## 0.1.2 - 2026-10-27

DeepSeek-V4.1-Flash LoRA recipes for the original and 1017 releases, loaded from DeepSeek's own weights, and multi-LoRA trainer scheduling for clients that train independently.

### Added

- `qwen3_4b_lora_32k`: Qwen3-4B LoRA on 4× H100 with 32K context, one model replica per GPU.
- DeepSeek-V4.1-Flash-1017 LoRA recipes, `deepseek_v41_flash_1017_lora_8k` and `deepseek_v41_flash_1017_lora_32k`, which load the 1017 release (#129).
- DeepSeek-V4.1-Flash LoRA recipe with 32K context, `deepseek_v41_flash_lora_32k`. It splits each layer across eight GPUs instead of four (#128).
- `deepseek_v41_flash_lora_8k` falls back to B200 and H200 for the trainer and samplers (#128).
- `trainer_max_clients_per_batch` caps how many multi-LoRA clients a trainer trains together. With `1`, each client's step runs on its own, so clients that train independently no longer wait for each other (#124).
- Samplers serve SGLang's Prometheus metrics at `/metrics`, so Modal collects inference metrics for sampler deployments (#125).
- Example launchers for the Qwen3.8-27B DAPO run on 4 H200s and 12 B200s (#135) and the Qwen3.5-9B SWE-Gym run on B300s (#136).

### Changed

- DeepSeek-V4.1-Flash recipes load DeepSeek's FP8/FP4 release instead of a BF16 export. The trainer dequantizes it to BF16 as it loads, and samplers serve it with FP8 routed experts. Redeploying downloads the release once (#128).
- `deepseek_v41_flash_lora_8k` keeps one sampler running for the length of a run, so the first sample doesn't wait for the weights to load. Recipes that subclass it inherit this (#134).
- DeepSeek-V4.1-Flash samplers queue up to 128 requests instead of 8, so a GRPO step's group samples no longer get 503s and retries (#141).
- Multi-LoRA trainers run optimizer steps, checkpoint saves and sampler-weight captures ahead of queued `forward_backward` requests, and serve queued training requests oldest first (#124).
- A sampling request delivered again with the same request ID shares the in-flight work or the completed response instead of generating twice (#120).

### Fixed

- DeepSeek-V4.1-Flash trainers on B200 no longer hang in their first step (#128).
- Miles' gloo process group waits as long as `distributed_timeout_minutes`, so trainers on slow-loading hosts no longer time out after 30 minutes (#128).
- The Miles v0.1.2 trainer image, the Miles Megatron base image and the SGLang image are pinned to exact commits or digests, so a cached build cannot serve an older fork branch (#128, #131).
- The rollout image builds SGLang's radix-tree binding from the pinned fork revision, so the scheduler no longer crashes when routing observations are enabled (#137).
- A sampling request that waited longer than 30 seconds, or a future polled after its sample completed, no longer fails with `unknown key family: sampler_warm` (#130).
- Training runs named in sampler paths, `<model_id>:train:0`, resolve to the model's checkpoints, so tinker-cookbook's `run_inspect_evals` finds the base model (#130).

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
