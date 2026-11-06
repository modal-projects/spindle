# Trainer runtime memory snapshots

**Result:** a separate four-H100 container restored the warmed 9B Miles trainer
with **GPU P2P/CUMEM NCCL**, matched the saved model/gradient/optimizer/RNG state,
passed all process-group checks and completed another backward/Adam update.
This uses explicit per-actor NCCL suspend/resume, `NCCL_CUMEM_ENABLE=1`, NVLS
disabled, and stable loopback addresses for Ray and communication bootstrap.
The first forward matches exactly. The numerical follow-up below audits tensor
storage byte for byte and isolates the training drift to nondeterministic
attention. The production-engine probe below also restored the backend HTTP
subprocess and served a new client through a freshly started Engine frontend.
With deterministic attention, a separate-container restore also
matched gradients, updated weights/optimizer tensors and log probabilities
exactly. Earlier socket-NCCL results are retained below.

This experiment asks whether a new container can resume an initialized Spindle
trainer from a Modal CPU/GPU memory snapshot, without loading model or optimizer
checkpoints. The first target is `Qwen/Qwen3.5-9B-Base`, Miles LoRA, TP=4 on
4 H100s in one container, using the existing 16K recipe and a 128-token probe.
It is an investigation, not a production trainer configuration.

## Production engine cold-start probe (2026-11-06)

`tests/manual/benchmark_qwen9b_engine_snapshot.py` extends the actor-level
experiment to the actual engine command path:

```text
HttpEngineClient -> TCP HTTP -> create_engine_app / Engine scheduler
  -> HttpBackendClient -> TCP HTTP -> spindle.engine.backend_http subprocess
  -> production miles_lora.build_executor / DistributedExecutor
  -> MilesRuntime -> Ray actors -> Megatron / GPU tensors / NCCL groups
```

The snapshot contains the backend subprocess, its HTTP server and thread/event
loops, Ray services/controller/actors, loaded base weights, six LoRA slots and
the initialized GPU runtime. It uses the unmodified production executor factory
and training endpoints. A diagnostic factory wrapper adds actor probes,
communicator suspend/resume and stable Ray addresses. The lightweight Engine
scheduler and its authenticated HTTP server start **after** restore, with fresh
frontend identity and no old client requests/futures.

The template uses the Qwen3.5-9B LoRA **16K configuration**, TP4, six rank-32
slots on four H100s, 16 CPUs and 64 GiB host memory. A **2,048-token** synthetic
forward/backward and Adam step warms the backend, after which that temporary
client is unloaded. The captured template has **zero registered training
clients**; the restored engine must accept a fresh client, reproduce its
pre-update forward, complete backward/Adam and unload it through the engine
API. Warmup at 2K does not establish performance for all sequence lengths up
to the configuration's 16K limit. Deterministic attention, CUMEM and disabled
NVLS retain the constraints of the preceding experiments.

Readiness means that a real authenticated HTTP request to
`/api/v1/models` succeeds on the new frontend. Communicator resume is included;
state fingerprints/group verification and training requests happen afterward.
The probe records client submission and readiness UTC timestamps as well as
monotonic durations within each container. Client-to-ready wall time includes
scheduling and platform restore, subject to clock synchronization; it is
separate from client-to-result time, which also includes verification and
training. A new task ID with the saved initialization ID/backend PID and
identical state fingerprints is required to call a run a restore.

**Separate-container engine restore passed.** Producer
`ta-01M6S3QDBMK3RAFT48SSAX5MWR` created initialization
`db3e6ceabd7b41e4b2cfda238af1de4d`; clone
`ta-01M6S49ARS73HKM3K1WEBA4F4R` retained that ID, the backend's independent
initialization ID and PID 20, while constructing a new Engine frontend.
Every saved model/gradient/RNG/process-group fingerprint matched before new
client work. All **48 NCCL and 10 Gloo groups** passed their collectives. This
empty template has no active client optimizer to resume; it creates one for
the newly registered client. Captured nonempty optimizer continuation was
validated separately in the numerical-parity probe below.

| Measurement | First confirmed cold clone |
| --- | ---: |
| Client submission to authenticated Engine HTTP readiness | 149.96 s |
| Submission to platform restore log (scheduling/setup) | 107.22 s |
| Platform restore log to Engine HTTP readiness | approximately 42.74 s |
| Recovery hook to Engine HTTP readiness | 5.19 s |
| NCCL resume within that hook | 5.11 s |
| Fresh-client registration | 0.243 s |
| First 2K forward (maximum logprob error 0.0) | 1.074 s |
| First 2K forward/backward | 0.741 s |
| Adam update | 0.029 s |
| Fingerprint/group verification plus API training after readiness | 8.86 s |

One optional confirmation call was cancelled while still queued for H100 capacity
(the platform warned at 19:13:00 UTC). It contributes no second latency sample.
The platform restore log has one-second precision; its intervals are approximate.
The **42.74 s includes** recovery of CPU/GPU/process-tree state, communicator
resume and starting the new frontend. The **5.19 s alone is not cold-start
latency**. The observed **149.96 s end-to-end** includes substantial waiting
before restore begins. These samples demonstrate reuse of the full warmed GPU
backend, not guaranteed scheduling latency or a production startup SLO.
The successful producer initialized and warmed from populated disk caches in
196.39 s before diagnostics/capture, so even that faster initialization was
substantially longer than the observed restore interval.

The normal-start controls reached a warmed, HTTP-ready engine in **264.95 s**
and **282.11 s** from submission. The second ran after the private kernel cache
had been populated; it still initialized a fresh backend. Backend initialization
alone took 117.95 / 178.83 s, and initialization plus forward/backward/Adam warmup
took 241.47 / 257.24 s. The harness additionally collected fingerprints before
starting the frontend (roughly 9 / 17 s); these diagnostics are included in the
264.95 / 282.11 s totals and would not be required in a production startup.
Fresh-client registration and the first warmed 2K training request succeeded
in both controls, with first-forward maximum error 0.0.

A short-lived variant successfully captured twice and resumed its producers,
but its three-call sequence and a delayed fourth call **all initialized fresh
backends**. Container arguments in the fourth call showed snapshotting enabled
with **no checkpoint assigned**. Platform logs showed capture and image save
completion; no cross-container restore was demonstrated by that variant. This
is not evidence that GPU state was restored incorrectly. The final harness
keeps snapshot producers alive for 120 s after HTTP readiness, like a serving
trainer, and the direct driver avoids creating an unrelated ephemeral App.
Producer-only publication wait is recorded separately and never applies to a
container with a different task ID. Attribution of the earlier misses to the
producer lifetime is a hypothesis unless isolated with further controls.

One earlier initialization failed before capture because the backend's chosen
loopback port was already in use when its server finally bound after model
loading. The normal launcher also releases its port probe before initialization;
this test leaves that production startup race unresolved. It is distinct from
snapshot capture and communicator recovery.

The private kernel cache uses the same compiler environment variables as the
production cache, mounted from a dedicated test Volume. This experiment does
not register instances in the production control plane, warm a sampler pool,
create public tunnels or run the external SDK gateway. Its engine HTTP server
listens on loopback; both HTTP hops use actual TCP sockets.

```bash
export SPINDLE_ENGINE_SNAPSHOT_APP=spindle-qwen9b-engine-snapshot-1106
uv run modal deploy tests/manual/benchmark_qwen9b_engine_snapshot.py
uv run python tests/manual/drive_qwen9b_engine_snapshot.py \
  --mode snapshot --repeats 3 --output /tmp/qwen9b-engine-snapshot.json
uv run python tests/manual/drive_qwen9b_engine_snapshot.py \
  --mode baseline --repeats 1 --output /tmp/qwen9b-engine-baseline.json
```

Results, identity checks, lifecycle observations and transport excerpts are in
[`engine-cold-start-results.json`](assets/trainer-memory-snapshots/engine-cold-start-results.json).
Exact live harness/diagnostic and production backend sources with SHA-256
manifests are in
[`engine-cold-start-sources.json`](assets/trainer-memory-snapshots/engine-cold-start-sources.json).
The CPU harness check exercises both TCP HTTP servers and the real Engine
scheduler with a test executor; 33 engine/backend HTTP tests and 59 Miles backend
tests also pass. This remains an isolated experiment: production trainer
Functions have not been converted to snapshot-backed classes.

## Exactly what was saved and restored

The experiment did **not** export a model-plus-runtime file to a `modal.Volume`
and then load that file. It used Modal's deployed Function/Cls memory-snapshot
lifecycle. Modal persisted the snapshot in platform-managed storage and selected
it when starting a compatible new container. The harness did not obtain a
portable snapshot file, snapshot ID, or user-controlled save/load path.

There were two separate kinds of persisted data:

| Storage | Contents | How the experiment used it |
| --- | --- | --- |
| Dedicated Modal Volume, mounted at `/assets` | Original Hugging Face `Qwen/Qwen3.5-9B-Base` files | Downloaded once by `prepare()` and read during the initial trainer construction |
| Modal-managed CPU/GPU memory snapshot | Warmed container process tree and CUDA state, including the already-loaded base-model shards, trained LoRA adapter and Adam state | Captured after `@modal.enter(snap=True)` and restored automatically for a later container |
| Local JSON evidence files | Hashes, group inventory, IDs, timings and metrics | Used to verify the experiment; these files cannot restore a trainer |

The capture and restore sequence was:

1. **Build the initial runtime.** In `@modal.enter(snap=True)`, construct a real
   `MilesCommandBackend`. This starts its background event-loop thread, local Ray
   cluster and four trainer actors, creates process groups, reads the base weights
   from `/assets`, and places the TP=4 shards on the GPUs.
2. **Materialize training state.** Create one rank-32 LoRA job, perform a synthetic
   128-token backward/Adam update, run a reference forward, and record per-rank
   state hashes and collective results. The adapter is already trained for one
   step at this boundary; the saved state is more than frozen base weights.
   In suspend mode, verify a communicator suspend/resume round trip, then suspend
   every registered NCCL group again before returning from the snapshot hook.
3. **Let Modal capture the container.** The class enables
   `enable_memory_snapshot=True` and
   `experimental_options={"enable_gpu_snapshot": True}`. After the snapshot hook
   returns, Modal handles CPU/process and CUDA checkpointing. Its documented GPU
   mechanism drains outstanding GPU work and moves GPU allocations/state through
   host memory for the container snapshot. The original container then resumes;
   passing checks in that original container alone does not establish cloning.
4. **Start a different container from that snapshot.** Invoke the same deployed
   class again after publication. On a compatible worker, Modal restores the
   process tree and CUDA state before `@modal.enter(snap=False)` runs. The
   `MilesCommandBackend` constructor, `ray.init()`, actor/model initialization,
   LoRA slot loading and warmup hook are skipped. Ray is retained, not restarted.
   In suspend mode, the post-restore hook resumes the saved NCCL backends on
   every actor before submitting any training work.
   The harness makes no application-level model or optimizer checkpoint-load call
   on this path.
5. **Verify the restored runtime.** Generate a fresh boot ID after restore, run
   the saved actors' collectives and hash comparisons, reproduce reference log
   probabilities, and execute the next backward/Adam update. The saved
   initialization ID and actor PIDs match, while the Modal task ID changes.

This is a reusable initialization snapshot associated with a deployed function,
its configuration and compatible worker hardware. It is not a demonstration of
periodically saving an arbitrary live training session to a volume and explicitly
restoring it by checkpoint path. The image and referenced asset volume remain
dependencies; the experiment did not test restoration after deleting them.

Only the trainer backend and its dependencies were instantiated. The Spindle
Engine HTTP frontend, tunnels, control-plane registration and sampler were not
started or restored. A production integration would need to initialize those
per-container services and identities after restoring the backend.

The quoted timings describe different intervals in the successful experiment:

- **333.51 s:** the original initialization hook, starting after Python imports
  and ending after model loading, first training step, reference forward and
  initial fingerprints. This is not just checkpoint-file I/O.
- **About 41 s:** the new container's `Restoring Function from memory snapshot.`
  log to its post-restore hook, using second-resolution logs. This is a combined
  platform resume interval, not a separately measured CPU, GPU or storage stage,
  and excludes earlier scheduling/queue time.
- **18.06 s:** the `verify()` method after restore, including full tensor hashes,
  collectives, forward comparisons and another training step. This validation
  overhead is not a required 18-second production startup phase.

## What must be captured

| State | Expected mechanism | Open compatibility question |
| --- | --- | --- |
| Sharded weights, buffers, gradients, Adam state | GPU allocations plus CPU process memory | Allocator and driver allocation types must support checkpoint/restore |
| CUDA contexts, streams, events, loaded kernels, captured graphs | CUDA checkpoint/restore | Driver, GPU topology, IPC and shared GPU mappings |
| PyTorch/Megatron process groups | CPU state plus all underlying NCCL/Gloo resources | Communicator objects alone do not restore IPC handles, NVLS multicast or RDMA resources |
| Ray driver, runtime loop, actors, GCS, raylet, object store | Entire container process tree, threads, shared memory and local connections | Whether all local connections, identities and timeouts survive cloning |
| Python/NumPy/PyTorch RNG and training counters | CPU/GPU memory | Clones intentionally inherit RNG state; new instance identities must be generated after restore |
| Engine HTTP server, Modal tunnels, control-plane registrations | Recreate after restore | External peers, URLs, tokens and leases belong to the new instance |
| Files and mounted volumes referenced by the runtime | Existing image/volume files | Memory snapshots do not version mutable external volumes |

The current trainer builder in `src/spindle/providers/modal/deployment_apps.py`
calls `run_trainer()` inside the function body. That starts the backend HTTP
subprocess and initializes Miles/Ray after Modal's initialization snapshot point.
Turning on memory snapshots there would not capture the initialized trainer.
The expensive trainer initialization must move into `@modal.enter(snap=True)`.

`MilesRuntime` owns a background asyncio thread, a local Ray cluster (for
single-node deployments), worker-manager and trainer handles, and an object store.
Retaining only the Python driver object cannot retain its dependencies. Restarting
Ray after restore would invalidate the existing actors and their resident state.

## Reproduction

The manual harness is `tests/manual/benchmark_qwen9b_memory_snapshot.py`; its
test-only actor subclass is `tests/manual/qwen9b_snapshot_probe.py`.
The app and asset volume use dedicated names and do not register a production
engine, start a sampler, or access production checkpoints.

For the successful configuration, from the repository root using the intended
Modal profile/environment:

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-memory-snapshot-loopback-1105
export SPINDLE_SNAPSHOT_TRANSPORT=socket
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py --mode prepare
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode baseline --repeats 1 --output /tmp/qwen9b-baseline.json
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --require-restore --output /tmp/qwen9b-snapshot.json
```

Use a distinct `SPINDLE_SNAPSHOT_APP` for independent experiments. Omitting the
transport/Ray overrides exercises the default configuration, which stalled
capture in the recorded runs. The script reports the snapshot creator task ID
and whether a result came from a different container; `--require-restore` fails
if all successful calls only initialized or resumed their original containers.
The historical JSON records predate these explicit fields, so their proof uses
the matching initialization IDs, differing task IDs and platform restore log.

The local entrypoint calls the **deployed** class with `Cls.from_name()`. The
warning about disabled snapshots in an ephemeral `modal run` app does not apply
to that deployed class. Snapshots are not exercised by invoking an ephemeral
class directly.

Each verification runs in a single-use container. Before snapshotting, the harness:

1. Initializes the real Miles backend, Ray actors and TP=4 model.
2. Loads a rank-32 adapter and runs backward plus Adam, materializing lazy state.
3. Records reference forward log probabilities.
4. Runs an all-reduce with a known answer on every existing process group.
5. Hashes all model parameters/buffers, optimizer state, and CPU/CUDA/Python/NumPy
   RNG state in each actor.

After restore, it repeats the group collectives and hashes, checks the reference
log probabilities, then requires another successful backward/Adam update and
changed weight and optimizer hashes. No model or optimizer checkpoint is loaded
on this path. A successful health response alone is not a pass.

An `initialization_id` is generated before capture and a `boot_id` after restore.
Multiple results sharing the former but having different boot/task IDs, together
with Modal's restore log, establish reuse of an actual snapshot. Otherwise a
successful result might be a cold initialization or snapshot fallback.

The first snapshot-building call includes capture overhead. `call_to_result_s`
includes scheduling and verification; `initialization_s` is the work bypassed by
restore; `verification_s` measures the `verify()` method. These are not isolated
platform restore latency measurements. Use Modal container startup metrics to
separate queueing, image startup, capture, and restore.
The harness waits 60 seconds between repeat calls for asynchronous snapshot
publication; this wait is outside `call_to_result_s`. A worker with a different
compatibility fingerprint may still need its own initial snapshot.

For a transport diagnostic, deploy a separate app with
`SPINDLE_SNAPSHOT_APP=spindle-qwen9b-memory-snapshot-socket-1105` and
`SPINDLE_SNAPSHOT_TRANSPORT=socket`, using those variables for subsequent commands
as well. This disables NCCL peer-to-peer, shared-memory, NVLS and IB transports
and confines sockets to loopback. Success there would establish a compatibility
workaround, not support for the default high-performance communication path.

`SPINDLE_SNAPSHOT_TRANSPORT=no-nvls` disables only NVLS and retains GPU P2P/IPC.
Use a separate `SPINDLE_SNAPSHOT_APP` for each variant to preserve the evidence
and avoid invalidating snapshots from another configuration.

For explicit communicator suspension, set `SPINDLE_SNAPSHOT_COMMS=suspend`.
After warmup and the initial collective probes, the driver stops submitting
training commands. Each actor barriers its NCCL groups in consistent order,
synchronizes CUDA, then calls `group._get_backend(device).suspend()` on each
initialized group, including the default group. PyTorch 2.13 exposes these
backend methods; they call `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)` and
`ncclCommResume(comm)` with NCCL 2.29.7 or newer. They are not the vLLM-specific
`suspend_device_comms()` / `resume_device_comms()` helpers.

The harness first performs a suspend/resume round trip without a snapshot and
checks all state fingerprints, group collectives and reference log probabilities.
It then suspends again for capture. The post-restore hook resumes every saved
backend before verification runs, in both the original and cloned containers.
This mode also selects loopback for NCCL/Gloo bootstrap sockets and explicitly
sets `NCCL_CUMEM_ENABLE=1`: Miles otherwise defaults it to `0`, which selects the
legacy P2P/IPC path rather than the P2P/CUMEM path used by the successful controls.
Selecting loopback does not disable GPU P2P communication.
Missing APIs and suspension failures raise errors instead of silently skipping
communicators. This mode only handles the idle, one-GPU-per-actor workload in
this harness; it does not coordinate arbitrary concurrent training submissions.

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-nccl-suspend-cumem-1106
export SPINDLE_SNAPSHOT_TRANSPORT=no-nvls
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
export SPINDLE_SNAPSHOT_COMMS=suspend
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --require-restore \
  --output /tmp/qwen9b-nccl-suspend.json
```

`tests/manual/benchmark_ray_memory_snapshot.py` is a CPU-only control using the
same Miles image. Deploy it before running it. It snapshots a local Ray cluster
and four stateful actors, then verifies actor identities and mutations after
restore. Its `--node-ip 127.0.0.1` option isolates the effect of Ray advertising a
container IP instead of loopback.

## Observed results

The baseline passed on 2026-11-05 using 4 NVIDIA H100 80GB HBM3 GPUs, driver
610.57.04, PyTorch 2.13.0+cu130, CUDA runtime 13.0, and NCCL 2.29.7.
The live experiment used source commit
`e29c8d5` (see the recorded repository revision
when reproducing); the PR adds the harness and evidence rather than modifying
the production trainer path.
The result is retained in
[`qwen9b-baseline.json`](assets/trainer-memory-snapshots/qwen9b-baseline.json).

- Initialization, slot loading, first backward/Adam, reference forward and initial
  fingerprints: **268.62 s**. First backward alone took **152.31 s**, including
  compilation and lazy initialization.
- Verification: **14.12 s**, including full model/optimizer/gradient fingerprints.
  The warmed second backward plus optimizer took about **0.57 s**.
- Client call to completed verification: **334.71 s**. This includes imports and
  container startup outside the initialization timer.
- All four sharded model hashes differed, as expected. Combined live allocations
  reported by PyTorch were **20.61 GiB**, excluding CUDA/NCCL allocations outside
  PyTorch's allocator.
- The collective checks covered **58 distinct groups: 48 NCCL and 10 Gloo**
  (22 memberships per rank). Log probabilities matched exactly; optimizer state
  and model hashes changed after the second successful update.

This baseline used container-local compilation caches, whereas production
`run_trainer()` mounts Spindle's persistent kernel cache. It establishes
correctness of the workload and probes, not a production startup speedup.

### Ray network identity

The CPU-only control reproduced a failure independently of CUDA/NCCL. Ray 2.58.0
created a valid CPU snapshot, but the next container's actor calls failed with
`ActorUnavailableError` / `No route to host`, targeting the original container's
`172.20.x.x` actor address. The restore log explicitly said
`Restoring Function from memory snapshot.`

Passing `_node_ip_address="127.0.0.1"` alone was insufficient: Ray's
`services.resolve_ip_for_localhost()` rewrote it to the external container IP.
The test-only workaround preserves loopback during that normalization and
asserts the actual `NodeManagerAddress` is `127.0.0.1`.

With that workaround, two independent restored containers successfully accessed
the same saved actor PIDs and counter values, then incremented each counter.
The saved snapshot's counters remained unchanged for the next clone. Measured
client call times were **5.85 s** and **7.38 s**, versus approximately **20–21 s**
for snapshot-creating calls. Results are in
[`ray-forced-loopback.json`](assets/trainer-memory-snapshots/ray-forced-loopback.json).
This demonstrates local Ray process-tree restoration, not reconnection to a
remote Ray cluster.

### GPU communication capture

The default 9B configuration reached the capture boundary after a successful
backward, Adam step and all group probes, then made no progress for **10 minutes**
before the experiment was stopped. Platform logs confirmed
`memory checkpoint starting; container liveness counting stopped`, with no
corresponding resume/save-complete event during that window. This is a bounded
capture stall, not a reported CUDA error or proof of one particular driver bug.

Disabling **only NVLS** still stalled for more than **5 minutes** at capture and
was stopped. Logs verified `NCCL_NVLS_ENABLE=0`, while NCCL channels still used
`P2P/IPC`. Consequently, disabling NVLS alone is not a sufficient workaround for
this trainer. The experiment does not isolate GPU IPC from all other shared
allocation mechanisms individually.

The socket-only NCCL configuration completed capture and resumed the original
container approximately **24 s** after the capture boundary. Full fingerprints,
log probabilities and another backward/Adam update passed. This initial capture
ran on **4 H200s**, because Modal upgraded the requested H100s. The harness pins
`H100!:4` for subsequent experiments to avoid mixing GPU types; override with
`SPINDLE_SNAPSHOT_GPU` if needed. The capture result is in
[`qwen9b-socket-capture.json`](assets/trainer-memory-snapshots/qwen9b-socket-capture.json).
It is not evidence of a separate restored container by itself.

### Explicit NCCL suspension (2026-11-06)

The follow-up uses PyTorch 2.13's backend `suspend()` / `resume()` methods on
every initialized NCCL group, with NVLS disabled and GPU P2P retained. A
pre-capture control checks that suspension alone preserves the initialized
trainer before testing platform checkpoint/restore.

The initial 9B run successfully suspended and resumed all **48 distinct NCCL
groups** (18 memberships per rank). All 58 NCCL/Gloo collective checks and
model/gradient/optimizer/RNG fingerprints matched, and the reference forward
had maximum log-probability error **0.0**. A second suspension completed in
roughly **17 ms per rank**, but capture then stalled for more than five minutes
and was cancelled. No post-capture hook or restored trainer ran in this attempt.
The transport logs show **P2P/IPC**, with `NCCL_CUMEM_ENABLE=0` set by Miles.
Evidence is in [`qwen9b-nccl-suspend.json`](assets/trainer-memory-snapshots/qwen9b-nccl-suspend.json).

Standalone subprocess controls separated NCCL behavior from the model and Ray:

| Control | Capture/restore result |
| --- | --- |
| 2 H100s, one NCCL group, NVLS off, P2P/CUMEM | Two different containers restored the saved worker PIDs and tensor sums, resumed NCCL, and passed all-reduce |
| 4 H100s, the trainer's 48 NCCL / 10 Gloo group definitions, NVLS off, P2P/CUMEM | A different container restored and passed collectives on every group; each rank resumed 18 NCCL memberships |
| 2 H100s, same single-group control with only `NCCL_CUMEM_ENABLE=0` added | Pre-capture suspend/resume passed, but capture stalled for more than two minutes and was cancelled |

The successful controls log **P2P/CUMEM**, not socket data transport. They
demonstrate functioning NCCL collectives after real snapshot restoration, but
only check a small tensor's sum and the saved processes; they do not contain a
model, gradients, optimizer, Ray cluster or training step. A second four-GPU
call initialized a new snapshot rather than restoring the first; the later
successful clone explicitly matches the first snapshot's initialization ID,
creator and worker state.

Control results and bounded transport excerpts are retained in
[`nccl-suspend-minimal.json`](assets/trainer-memory-snapshots/nccl-suspend-minimal.json),
[`nccl-suspend-groups.json`](assets/trainer-memory-snapshots/nccl-suspend-groups.json)
and [`nccl-suspend-legacy.json`](assets/trainer-memory-snapshots/nccl-suspend-legacy.json).
The exact diagnostic scripts and group layout are archived in
[`nccl-suspend-control-sources.json`](assets/trainer-memory-snapshots/nccl-suspend-control-sources.json);
extract its `source_files` under `/tmp` and run the corresponding Modal deploy/run
commands with the repository's `src` on `PYTHONPATH` to reproduce them.

### Full trainer restore with P2P/CUMEM (2026-11-06)

With `NCCL_CUMEM_ENABLE=1`, the full 9B trainer captured successfully and a
**different four-H100 container restored it**. The existing communicator objects
were retained; the hooks suspended and resumed their NCCL-managed resources.
NCCL channel logs show GPU **P2P/CUMEM**. The successful settings retain GPU P2P
and shared-memory transport support while disabling NVLS; no throughput
comparison against normal NCCL or the socket workaround was performed.

| Check | Result |
| --- | --- |
| Snapshot initialization ID | `5cf46e87aebe4de0b096c607364b4dae` in producer and clone |
| Producer container | `ta-01M6RRFDC2B3G7FKX5PBKR5SER` |
| Restored container | `ta-01M6RRVVXD6Q07N9H3QRV6S8FR` |
| Saved actor PIDs, parameters/buffers/gradients, Adam and RNG fingerprints | Exact match |
| Group checks after resume | All 48 NCCL / 10 Gloo definitions passed; 18 NCCL memberships resumed per rank |
| Reference forward | Maximum log-probability error **0.0** |
| Continued training | Backward/Adam succeeded; optimizer step advanced from 1 to 2; weights and optimizer changed |
| Producer initialization, after imports and before suspension control | **254.38 s** |
| Platform restore log to post-restore hook | Approximately **13 s**, excluding scheduling |
| NCCL resume on restored actors | Approximately **1.71 s per rank**, running concurrently |
| Verification after communicator resume | **12.79 s**, including hashes, all groups and another training step |

The platform restore log appeared at 15:50:32 UTC, the post-restore hook at
15:50:45 UTC, and communicator recovery completed at 15:50:47 UTC. These
second-resolution logs measure combined platform stages; they are not isolated
storage/GPU timings or a production persistent-cache benchmark. The earlier
snapshot-building run took 302.43 s to initialize. Another call waited for H100
capacity and built the second snapshot; only the third call actually restored.

**Initial numerical continuation check.** The clone started from
identical saved state and reproduced the first forward exactly. After the next
update, model and optimizer hashes differed from the producer's; RNG hashes
still matched. The maximum post-update log-probability difference was
**0.2948008**, and gradient norms were 570.7933 versus 570.8847. Two independent
fresh initializations also differed in model/optimizer fingerprints before
capture and in their reference outputs (maximum difference 0.1976738), despite
matching RNG fingerprints. That shows variation already exists before cloning;
it does not establish that the restored continuation's numerical differences
are acceptable or entirely explained by ordinary nondeterminism. The follow-up
below isolates the cause with same-state replay and an attention-backend
intervention.

Raw results, per-rank suspend/resume inventories, reference/post-update outputs,
transport excerpts, source hashes and continuation comparisons are in
[`qwen9b-nccl-cumem-restored.json`](assets/trainer-memory-snapshots/qwen9b-nccl-cumem-restored.json).
This establishes a functional, single-container, TP=4 LoRA trainer restored with
GPU P2P communication. NVLS-enabled snapshots, multi-node/RDMA restoration,
full-parameter optimizer state and production frontend integration remain
untested. All experiment apps were stopped afterward.

### Byte-level audit and numerical cause (2026-11-06 follow-up)

The original 0.2948008 post-update log-probability difference warranted a
stronger check than aggregate state fingerprints. The parity probe in
`tests/manual/qwen9b_snapshot_parity.py` adds two independent diagnostics:

1. **Direct byte comparison.** Before capture, retain independent CPU `uint8`
   copies of each unique live CUDA tensor storage. Enumerate named parameters,
   registered buffers, gradients, optimizer state and FP32 masters explicitly;
   also traverse model/DDP/optimizer objects and GC-visible physical CUDA
   tensors to cover flat buffers and other live tensor workspaces. Compare the
   entire storage, including its padding and views, after NCCL recovery and
   before any model work. This uses `torch.equal` on bytes, not a floating-point
   tolerance. No tensor subclasses were skipped in these runs.
2. **Same-state replay without snapshotting.** Restore saved tensor bytes,
   mutable model/optimizer bookkeeping and CPU/CUDA/Python/NumPy/Megatron RNG
   state in place, retaining the same objects and communicator references.
   Repeat forward/backward/Adam from the same state three times before capture.
   Record gradient comparisons before Adam, then updated model/optimizer values
   and output log probabilities. A CPU Adam replay check also validates that
   the replay helper restores moments, step counters, weights and gradients.

The default-attention experiment restored into container
`ta-01M6RVXBP00M69N0XM7XJGEBYR` from producer
`ta-01M6RV9C723B1MCMT6C0P344QR`. Every rank passed the direct byte comparison:
**5,465,584,916 bytes per rank, 21,862,339,664 bytes total**, with no mismatches.
The audit covered 3,556 explicitly named tensor entries and 2,350 unique CUDA
storages per rank, including optimizer masters/moments and DDP storage. These
counts include aliases among named tensors; underlying storages are deduplicated.
The saved inventories and pre-capture control results also matched the producer.
CUDA allocations held only inside native libraries are outside this tensor
enumeration; the audit does not claim to read every byte of the CUDA driver.

The large numerical drift reproduced **before any platform snapshot**: each
replay restored the same bytes and reproduced the pre-step forward exactly, but
gradients differed before Adam. The first producer's later replays had maximum
post-update log-probability differences of **0.2770133** and **0.1896734**.
Thus the snapshot is not necessary to produce the observed effect.

The isolated attention microtest used the installed FlashAttention APIs on an
H100, BF16, a 128-token sequence, four query heads, one KV head and head dimension
256, matching this trainer's TP-local full-attention shape. Repeating identical
inputs showed varying K/V gradients under **FlashAttention 3.0.0**. In the
20-repeat no-KV-sharing control (four KV heads), its results were identical;
FlashAttention 2 was identical with and without sharing. FlashAttention 3's
deterministic backward explicitly rejected head dimension 256. The separately
tested FlashQLA gated-attention case was identical in all 32 repetitions, with
its internal parallelism on/off and temporary allocations zeroed/not zeroed.

The targeted full-model intervention is:

```bash
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
```

Set this **before constructing the producer runtime** and preserve the same
configuration after restore. In this image, Transformer Engine selects
**FlashAttention 2.7.4.post1** for deterministic attention at head dimension 256.
This is an official [Transformer Engine determinism control](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/api/pytorch.html),
not a replacement for the NCCL snapshot hooks. No NCCL settings change between
the numerical controls.

In one initialized model, three deterministic-attention replays had **identical
gradients, updated model weights, optimizer tensors and log probabilities**.
Toggling only the attention modules' determinism flags back off selected
FlashAttention 3.0.0 and restored the variation: the second FA3 replay differed
from the first by **0.1744671** in post-update log probability. Its model weights
differed by up to **0.0001831**; gradient sign changes and Adam updates make a
small backward discrepancy visible in the final outputs. The replay starts
from the same saved model/RNG state for each trial.

The deterministic configuration also completed a **separate-container restore**:
producer `ta-01M6RW3FRE86N7NG88KJGFPV7R` to clone
`ta-01M6RWPSBPXHS78XX7QYT3CRSR`. The clone audited **21,862,339,760 CUDA
storage bytes across four ranks**, with zero mismatches. Both post-restore
training replays matched the producer's reference gradients, updated model
weights, optimizer tensors and log probabilities exactly: **zero differing
elements and maximum log-probability difference 0.0**. The shared initialization
ID and saved inventories/control results establish that this was a restored
runtime, rather than a fresh initialization. Each configuration produced two
compatibility snapshots before the third invocation restored a different container.

Comparisons are made **within each backend**, not between FA2 and FA3. Switching
the backend also changes forward rounding: the first FA3 forward differed from
the FA2 reference by 0.1203804 in that run. That cross-backend difference is not
a restore error and is not the 0.1744671 same-backend replay measurement.

Enable the byte audit and replay controls using:

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-snapshot-parity-det-1106
export SPINDLE_SNAPSHOT_PARITY=1
export SPINDLE_SNAPSHOT_TE_DETERMINISTIC=1
export SPINDLE_SNAPSHOT_TRANSPORT=no-nvls
export SPINDLE_SNAPSHOT_COMMS=suspend
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --require-restore \
  --output /tmp/qwen9b-snapshot-parity-det.json
```

Omit `SPINDLE_SNAPSHOT_TE_DETERMINISTIC` (or set it to `0`) to reproduce the
default-attention controls. The deterministic run also temporarily reenables
FA3 for its paired counterfactual, then returns to deterministic attention before
capture. This diagnostic retains large CPU copies and extra tensor storage;
its memory use and timings are not production startup measurements. No
throughput cost of deterministic attention was benchmarked.

The reviewable measurements are in
[`numerical-parity-summary.json`](assets/trainer-memory-snapshots/numerical-parity-summary.json).
Full per-storage manifests, all comparisons and exact live diagnostic sources
are retained in the gzip-compressed JSON archive
[`numerical-parity-results.json.gz`](assets/trainer-memory-snapshots/numerical-parity-results.json.gz).
The summary records compressed/uncompressed SHA-256 checksums. Decode with
Python's `gzip.open(..., 'rt')` and `json.load()`; the archive contains hashes and
metadata, not model tensor values.

### Full trainer restore with socket NCCL (2026-11-05)

The combined configuration passed in a **separate container on four H100s**:

| Check | Result |
| --- | --- |
| Saved initialization ID | `ee53ae3787f2494b9ed13665ab3ba1c4` in both containers |
| Original container | `ta-01M6Q0MPTF3FAA5YBASHBJ2WTR` |
| Restored container | `ta-01M6Q1QWZF3ZQMXNVGESBM688R` |
| Worker processes and group membership | Same saved PIDs and 58 group definitions; every collective passed |
| Restored parameters, buffers, gradients, Adam state and RNG | Exact per-rank hash match |
| Reference forward log probabilities | Maximum absolute error **0.0** |
| Continued training | Backward/Adam succeeded; optimizer step advanced from 1 to 2; model and optimizer hashes changed |
| Saved initialization and warmup work | **333.51 s**, excluding initial imports |
| Restore log to post-restore hook | Approximately **41 s** |
| Post-restore verification | **18.06 s**, including full tensor hashes and continued training |

The restored container logged `Restoring Function from memory snapshot.` at
23:47:11 UTC and entered the post-restore hook at 23:47:52 UTC. It did not execute
the trainer initialization hook. Evidence, complete rank inventories and hashes
are retained in [`qwen9b-restored.json`](assets/trainer-memory-snapshots/qwen9b-restored.json).
That call's client-to-result duration includes waiting behind other startup work
and must not be used as an isolated restore latency.

**These initial socket results did not establish continuation parity.** The first forward and all saved state
matched exactly, but the original container's and restored container's subsequent
updates were not byte-identical. Gradient norms were 572.6663 and 572.5220 (about
0.025% apart); post-update model and optimizer hashes differed on all ranks,
while RNG hashes remained equal. The experiment establishes a functional restored
trainer, not bitwise-equivalent training continuation. A repeated non-snapshot
control and numerical parameter/logit comparisons are needed to distinguish
normal backward/reduction nondeterminism from a restore-specific effect. The
harness now returns post-update log probabilities for that follow-up comparison.

The observed ~41 s resume versus ~334 s initialization supports the cold-start
idea for this probe. It is not a production speedup claim: the baseline used local
compilation caches, and production persistent kernel caches will change that
comparison. Socket NCCL also has a different training-throughput tradeoff from
the normal GPU P2P/NVLS configuration.

Only a short 128-token workload, rank-32 LoRA and an idle single-node capture
boundary were exercised. Engine HTTP serving, sampler publication, full-parameter
Adam state, long-context shapes and multi-node restoration were not validated.
The test apps and their remaining containers were stopped after verification;
the dedicated weights volume remains available for reproduction.

For the final 9B experiment combining the Ray fix and socket NCCL:

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-memory-snapshot-loopback-1105
export SPINDLE_SNAPSHOT_TRANSPORT=socket
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --output /tmp/qwen9b-loopback.json
```

## Platform evidence and interpretation

As checked on 2026-11-05:

- [Modal's memory snapshot guide](https://modal.com/docs/guide/memory-snapshots)
  describes multi-GPU programs as generally incompatible and warns that
  snapshots do not inherently improve storage-bound model loading. Expected
  wins are skipped initialization, conversion, compilation and warmup.
- [Modal's GPU snapshot implementation overview](https://modal.com/blog/gpu-mem-snapshots)
  describes capturing VRAM, CUDA objects, kernels and memory mappings after
  draining GPU work. It does not mean an in-flight kernel resumes mid-execution.
- [NVIDIA's cuda-checkpoint documentation](https://github.com/NVIDIA/cuda-checkpoint)
  distinguishes driver generations. Version 610 adds a coordinated job mechanism
  for `cuIpcGetMemHandle` IPC, while the documented limitations still include
  UVM and IPC using `cuMemExportToShareableHandle`. The actual Modal driver matters.
- [NCCL's communicator APIs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
  expose suspend/resume. Spindle currently has no lifecycle calls to suspend and
  resume every NCCL communicator before/after a memory snapshot. Installing a
  newer NCCL library alone does not add that integration to PyTorch.
- [NVIDIA NCCL issue 2337](https://github.com/NVIDIA/nccl/issues/2337) documents
  the NVLS multicast resource problem for CUDA checkpoint/restore. It remained
  open when checked. This is a specific compatibility concern, not proof that
  every Modal multi-GPU workload fails for that reason.

The first scope is an idle, warmed, single-container trainer. Restoring a
multi-node trainer adds coordinated capture of all containers, remote transport
and membership restoration, and rendezvous across new node addresses. Restoring
an arbitrary active training job also requires a consistent boundary for pending
commands, accumulated gradients, publication and durable control-plane state.
Modal startup snapshots are initialization templates, not automatic periodic
checkpoints of the latest training step.
