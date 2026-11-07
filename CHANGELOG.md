# Changelog

Notable changes to Spindle, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [semantic versioning](https://semver.org/). Before 1.0, a minor release (0.x.0) can change config defaults or deployment behavior; a patch release (0.x.y) only fixes bugs. Each release is tagged `vX.Y.Z` and published as a GitHub Release.

## Unreleased

### Fixed

- Sampling recovers when SGLang loads an adapter but its acknowledgement is lost. A later load error is accepted only when SGLang's registry confirms the same adapter name and checkpoint path.

## 0.1.6 - 2026-11-06

Add and remove configs on a running frontend without redeploying it, wait in line for a trainer slot instead of failing, and see each recipe's training queue.

### Added

- `spindle deploy` registers every recipe in one per-frontend registry instead of baking a configs file into the frontend image, so a deploy adds or updates configs instead of replacing the set; upgrading a pre-registry frontend carries over every recipe it serves (#181).
- `POST /api/v1/spindle/deployments` registers a config on a running frontend and deploys its apps, and `DELETE /api/v1/spindle/deployments/{name}` removes one (#160).
- `spindle add NAME --base BASE --FIELD=VALUE ...` registers a config through that API and waits until its apps are ready; `spindle remove NAME` removes a config while checking whether there are any models still training on it live (#160).
- `qwen25_7b_instruct_lora_32k`: Qwen2.5-7B-Instruct LoRA with 32K context, trained on 4× H100 and sampled on one GPU per replica.
- `qwen25_3b_lora_32k`: Qwen2.5-3B LoRA with 32K context, with one model replica per GPU. The shared server serves both Qwen2.5 recipes (#206).
- `GET /api/v1/spindle/queues` describes each recipe's training queue: waiting clients, free trainer slots, and how many clients were added, placed and released in the last hour (#201, #205).
- `spindle stop` stops a deployment: its frontend, then every trainer, inference, and sampler app it created. Pass the same `--app` and `--env` as the deploy.

### Changed

- Every app a deployment creates now starts with its frontend name, so `spindle deploy --app my-spindle` names its trainers `my-spindle-trainer-<recipe>` and its sampler pools `my-spindle-lora-<recipe>`. Two deployments with different `--app` values can share one Modal environment and recipe names. Deployments without `--app` keep their `spindle-...` names. A deployment made with `--app` on an earlier release gets new worker apps on its next deploy; stop its old `spindle-...` apps with `modal app stop`.
- When every trainer for a recipe is full, creating a model waits for a slot instead of failing with `trainer capacity exhausted`. The Tinker SDK shows the request's place in line. When a slot frees up, the request that has waited longest gets it. Rarely, a brand-new request takes it first.
- Spindle requires Modal 1.6.3 or newer, the first release that builds images with [image builder version](https://modal.com/docs/guide/images#image-builder-updates) 2026.10. Multi-node trainers use `modal.clustered` instead of the deprecated `modal.experimental.clustered`.

### Fixed

- The `qwen35_4b_lora_32k` and `qwen35_2b_lora_32k` trainers no longer run out of GPU memory on batches of many short sequences. They pack up to 16K tokens per GPU instead of 32K; sequences up to the full 32K context still train (#203).
- Full fine-tune training clients and `sample_with_replay` work with Tinker SDK 0.38 (#212).

## 0.1.5 - 2026-11-02

Faster DeepSeek-V4.1-Flash training, 64K and 128K DeepSeek-V4.1-Flash recipes, and frontends that can require a Modal proxy token.

### Added

- `deepseek_v41_flash_lora_64k`: DeepSeek-V4.1-Flash LoRA with 64K context. It trains at TP8 on B300 or B200; H200 cannot hold a 64K training sequence for this model (#178).
- `deepseek_v41_flash_lora_128k`: DeepSeek-V4.1-Flash LoRA with 128K context (#172).
- LoRA trainers can split one long sequence across GPUs for models that need it, such as DeepSeek-V4.1 (Miles `allgather_cp` with `qkv_format: bshd`), including batches that train several LoRA adapters at once (#172).
- `platform.url_label` sets the part of the frontend URL after the double dash, so `"prod"` gives `https://<workspace>-<environment>--prod.modal.run`. `platform.requires_proxy_auth` makes Modal turn away frontend requests without a [proxy token](https://modal.com/docs/guide/webhook-proxy-auth). Deployments that set neither keep their URL and stay reachable by stock Tinker clients (#180).
- `examples/shared-server`: one deployment a team shares, with one LoRA recipe for each bundled base model on a `prod` frontend in its own environment, a smoke test, a script that adds the proxy token to any Tinker script, and a workflow that redeploys it from `main` (#139, #189, #192).
- Every Spindle deployment is tagged with its version, such as `v0.1.5`. The Tag column of each app's deployment history shows which Spindle release it runs (#179).

### Changed

- When a frontend serves several recipes for the same model, requests by model name go to the recipe with the longest context, instead of whichever was deployed first. Pass a recipe's name as `base_model` to pick that recipe. `get_server_capabilities` lists every recipe and its context length. Sampling a saved adapter uses the recipe that trained it (#171).
- Deployments now download base weights for up to four recipes at once, so a small model no longer waits behind a large one (#182).
- The DeepSeek-V4.1-Flash 8K and 32K LoRA trainers now train several sequences together instead of one at a time, so training steps take less time and use more GPU memory. On two 8× H200 nodes, a 32K GRPO training step drops from 128 to 57 seconds. Recipes based on them inherit this; set `miles_cfg.dynamic_batch_size: False` and `miles_cfg.cli_options.qkv_format: "bshd"` to keep the previous behavior (#193).

### Fixed

- Scripts on Tinker SDK 0.36 or newer, including tinker-cookbook, no longer fail with a 404 when they exit. Spindle now accepts the SDK's end-of-session call and frees that session's models right away (#190).
- Creating a model no longer fails with `trainer capacity exhausted` when a trainer finishes starting while the model is being placed (#185).

## 0.1.4 - 2026-10-29

Faster DeepSeek-V4.1-Flash sampling and sampler starts, and agent skills for working with Spindle.

### Added

- `deepseek_v41_flash_lora_32k_b300`: the 32K DeepSeek-V4.1-Flash recipe on B300, which holds a 32K row at TP2 (#163).
- `spindle skills install` installs agent skills that explain deploying Spindle, where it differs from Tinker, and how to run experiments on it (#140).

### Changed

- The DeepSeek-V4.1-Flash 32K recipes train at TP4 instead of TP8 on H200 and B200, so each rank runs half as many micro-batches per step. The 1017 32K recipe inherits this (#163).
- The DeepSeek-V4.1-Flash 32K recipes decode with the checkpoint's draft heads (SGLang's DSpark mode). A long sample that used to finish alone at 30 to 60 tokens per second now finishes several times faster, and samples keep the model's own distribution because the full model checks every draft token. The sampler takes four to seven minutes longer to start, which a 32K run earns back within about five steps. The 8K recipes are unchanged (#165).
- DeepSeek-V4.1-Flash samplers run 128 requests at once instead of 32, so a GRPO step's samples generate together instead of in four waves (#147).
- DeepSeek-V4.1-Flash samplers start sooner: they read the checkpoint with eight threads instead of one (#148), each GPU rank pins its own eighth of the Engram tables instead of all ranks pinning one shared copy, which saves 200 to 250 seconds on 8× H200 (#149), and they keep their compiled kernels on the shared kernel-cache volume, like trainers, which saves about five minutes once the cache is warm (#146, #167). Other recipes can keep SGLang's caches on the volume too by setting `SGLANG_CACHE_DIR` under `/root/.cache/kernel-cache` in `inference_env`.
- Sampler images no longer build SGLang's prompt-indexing extension at startup; it ships prebuilt (#146).
- Every sampler sets `SGLANG_ENABLE_LORA_CSGMV_SPLIT_K=1`. SGLang still applies split-K only to the BF16 rank-32 CSGMV shapes it measured, and never in deterministic mode. LoRA servers already set the flag (#123); set it to `0` in `inference_env` to turn it off. `qwen35_9b_instruct_lora_256k` set the flag under its old name, so split-K never ran there before (#151).

## 0.1.3 - 2026-10-28

Faster training for LoRA classifiers that score several candidate labels per position, and cache-aware routing for LoRA samplers.

### Added

- `qwen3_4b_lora_32k`: Qwen3-4B LoRA on 4× H100 with 32K context, one model replica per GPU (#142).
- Qwen3.8-27B LoRA recipe with B200 samplers, `qwen38_27b_lora_16k_b200`, which serves with the TensorRT-LLM attention backend and 64-token pages (#145).
- LoRA samplers can route each request to the replica with its longest cached prefix with `inference_lora_routing = "cache_wait"`, while keeping request load and pinned KV memory within bounds. The router runs in Rust inside each sampling container. It measures the pool's prefill rate from completed requests unless `inference_routing_prefill_tokens_per_second` sets it, and `inference_routing_balance_absolute`, `inference_routing_balance_relative` and `inference_routing_cache_threshold` tune it. Pools without routable replicas, including pools scaled to zero and pools whose SGLang cache publishes no observations, fall back to Modal's gateway routing (#121, #122, #123).

### Changed

- LoRA recipes use `cache_wait` routing by default when SGLang runs one data-parallel rank, and Modal session affinity otherwise. `inference_lora_routing = "affinity"` keeps session affinity. `inference_routing_balance_absolute` defaults to 40% of `max_running_requests`, and at least 4 (#123).
- LoRA samplers enable SGLang's CSGMV split-K kernel. For rank-32 adapters on models with 4,096-wide hidden layers, such as Qwen3.5-9B and Qwen3-8B, it speeds up the LoRA kernels for decode-sized batches by up to 2.9× on H200 and 2.6× on B200. Set `SGLANG_ENABLE_LORA_CSGMV_SPLIT_K=0` in `inference_env` to disable it (#123).
- The frontend server runs in `platform.modal.region`, or `us-east` when no region is set. Only the CPU frontend is placed; trainers and samplers stay unpinned unless the recipe sets a region. Before, frontend containers could start in any region, and one on another continent added 2 s or more to each training step (#152).
- `forward` requests use the same binary transport and protobuf results as `forward_backward`, and queued forwards are batched into one trainer call (#153).
- The Miles backend prepares labels and weights with numpy instead of per-value Python loops (#154), and passes training rows and log probabilities to its GPU workers as arrays instead of Python lists (#158). Large batches spend less time between the request and the GPU work.
- When every datum in a batch has the same number of candidate labels (`target_tokens` shaped `[sequence_length, K]`), the Miles backend scores all K labels from one softmax over the vocabulary instead of K (#157).

## 0.1.2 - 2026-10-27

DeepSeek-V4.1-Flash LoRA recipes for the original and 1017 releases, loaded from DeepSeek's own weights, and multi-LoRA trainer scheduling for clients that train independently.

### Added

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
