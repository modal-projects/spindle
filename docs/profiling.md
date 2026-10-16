# Profiling a training step

Spindle can record a [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html)
trace of one training step on the Miles LoRA trainer. The trace shows, on a
timeline, what the GPU and CPU were doing during that step: the forward/backward
pass, the optimizer update, the LoRA weight export, and the gaps in between.
It is the fastest way to answer "where does my step time go?"

Profiling is off by default and has no overhead until you turn it on.

## 1. Enable it

Pick the step you want to trace and set one environment variable in the shell
you deploy from. The trainer inherits it:

```bash
SPINDLE_TORCH_PROFILE_STEP=2 uv run spindle deploy deployment.py
```

Then run your training loop as usual. Steps are counted from 0, so `2` traces
the third optimizer step. Use a small step index (1 or 2) rather than 0, so
warm-up work such as compiling kernels is not in the trace.

Optional knobs:

| Variable | Default | What it does |
| --- | --- | --- |
| `SPINDLE_TORCH_PROFILE_STEP` | unset (off) | Which step to trace. Setting it turns profiling on. |
| `SPINDLE_TORCH_PROFILE_RANKS` | `0` | Which trainer GPUs (ranks) to trace, e.g. `0,4` or `all`. Each trace is ~250 MB, so start with one. |
| `SPINDLE_TORCH_PROFILE_DIR` | `/checkpoints/torch-profile/<definition id>` on the checkpoint volume | Where trace files are written. |

Deploy without `SPINDLE_TORCH_PROFILE_STEP` to turn profiling off again.

## 2. What gets captured

- **One optimizer step.** Recording starts when the first `forward_backward`
  of the chosen step arrives at the trainer and stops right before the next
  step begins, so the trace covers all of that step's forward/backward calls,
  the `optim_step`, and the LoRA weight export that follows.
- **Rank 0 only, by default.** Trainers run one process per GPU; the trace is
  per process. Rank 0 is representative for data-parallel training; trace more
  ranks when you suspect imbalance (e.g. across tensor-parallel or context-parallel
  groups).
- **CPU and GPU activity**, with Spindle's phases labelled `spindle/forward_backward`,
  `spindle/optim_step`, `spindle/forward_only`, and `spindle/export_slot_peft` so you
  can find them quickly among the kernel names.
- A separate, CPU-only trace of the trainer's controller process, which covers
  the post-step sampler-weight save and publish.

Output files, per traced rank:

- `rank<N>.trace.json.gz` — the timeline, in Chrome trace format.
- `rank<N>.key_averages.txt` — a plain-text table of the top operators by time,
  handy for a first glance without a UI.

plus `controller.trace.json.gz` / `controller.key_averages.txt` for the
controller process.

## 3. View the trace in Perfetto

1. Download the trace from the checkpoint volume:

   ```bash
   uv run modal volume get spindle-checkpoints \
     /torch-profile/<definition id>/rank0.trace.json.gz .
   ```

   Or attach it to your W&B run so it lives next to the metrics (it appears
   under the run's **Files** tab as `profiler/rank0.trace.json.gz`):

   ```bash
   uv run scripts/upload_torch_profile.py --dir <trace dir> \
     --entity <entity> --project <project> --run-id <wandb run id>
   ```

2. Open <https://ui.perfetto.dev>, click **Open trace file**, and pick the
   `.json.gz` file (no need to decompress it). Everything runs locally in your
   browser.

3. Reading the timeline:
   - The top rows are **CPU threads** (Python and the PyTorch dispatcher); the
     rows named `stream N` are **GPU streams**, where kernels actually run.
   - Press `W`/`S` to zoom and `A`/`D` to pan. Type `spindle/` in the search box
     (`Ctrl+F` / `Cmd+F`) to jump between Spindle's phases.
   - Click a slice to see its duration and, for GPU kernels, the CPU call that
     launched it. Select a range (drag on the timeline) to get a time breakdown
     of everything inside it.

What to look for:

- **Idle GPU streams** inside `spindle/forward_backward`: gaps mean the GPU is
  waiting on the CPU (data prep, Python overhead) or on communication.
- **Long communication kernels** (`ncclKernel_*`, all-gather / reduce-scatter):
  the cost of tensor/context/data parallelism.
- **Time outside the labelled phases**: work between steps that is not
  training, such as weight export or checkpoint writes.

## 4. Where a cold start spends its time

A step trace explains steady-state speed. A cold start is different: the first
`forward_backward` on a fresh deployment waits for a GPU to be scheduled, the
container to boot, the model to load, and kernels to compile. Spindle times each
of these phases on every container, with no configuration needed, and prints one
JSON line per phase boundary to the container's log:

```json
{"event": "spindle_startup_mark", "component": "trainer-backend", "mark": "trainer_actors_initialized", "ts": 1792012345.1, "since_start_s": 212.4, "phase_s": 148.9}
```

- `since_start_s` is the time since that container's process started.
- `phase_s` is the time since the previous mark in the same component, so it is
  the cost of the phase that ends at this mark.

Trainer marks, in order (`component` in parentheses):

| Mark | Phase that ends here |
| --- | --- |
| `trainer_function_entered` (trainer) | Container boot and Modal runtime start. |
| `assets_reloaded` / `ray_cluster_joined` (trainer) | Volume reload, or Ray cluster formation for multi-node trainers. |
| `backend_spawned`, `engine_record_starting` (trainer) | Backend process launched. `engine_record_starting` carries `queue_s`, the wait between the control plane requesting the trainer and the container's process starting. |
| `backend_imports_done` (trainer-backend) | Importing Torch, Megatron and Miles. |
| `miles_args_parsed`, `ray_started`, `ray_workers_launched` (trainer-backend) | Ray startup and per-GPU worker actors. |
| `actor_model_loaded` (trainer-rank0) | Rank 0 built the Megatron model and loaded the base weights from the assets volume. |
| `trainer_actors_initialized`, `miles_backend_ready` (trainer-backend) | All ranks finished loading. |
| `backend_ready`, `engine_serving` (trainer) | The engine answered its health check and is reachable; `since_spawn_s` is the total from request to ready. |
| `first_slot_loaded`, `first_forward_backward_done`, `first_optim_step_done`, `first_sampler_export_done` (trainer-backend) | The first client's work, including kernel compilation. `duration_s` is the call itself. |

Sampler replica marks: `server_enter`, `sglang_healthy` (base model loaded and
CUDA graphs captured), `sidecar_healthy`, then `first_adapter_loaded`
(`sampler-sidecar`) when the first LoRA adapter is applied. The one-time pool
deployment prints `pool_deployed` (`sampler-pool`), and the model asset
download prints `assets_ready` (`model-assets`).

The control plane also keeps `spawned_at`, `started_at`, `ready_at` and the
phase breakdown on each trainer instance record, next to its state.

To list a deployment's marks as a table:

```bash
uv run modal app logs spindle-trainer-<config> \
  | grep -o '{"event": "spindle_startup_mark".*}' \
  | jq -r '[.component, .mark, .since_start_s, .phase_s] | @tsv'
```
