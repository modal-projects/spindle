# LoRA-aware rollout routing

LoRA sampling workers choose replicas locally. Each worker maintains a periodically
refreshed snapshot of the pool and counts its own in-flight generations. There is
no shared scheduler, distributed lock, or routing RPC on a warm request's path.

`inference_lora_routing = "affinity"` enables routing; `"naive"` keeps the existing
adapter-independent Flash behavior for comparisons. Base-model sampling and FFT
routing keep their existing paths.

## Placement and load

The router uses rendezvous hashing of the model ID to establish a stable ordering
of replicas. Successive versions of an adapter therefore prefer the same homes.
Within a set of warm replicas, request/conversation affinity breaks ties.

For each replica, estimated load adds new local reservations to reported active
requests, subtracting reservations already present at observation time to avoid
counting them twice. It is at least the local outstanding count. A replica is eligible when its load is
below `1.25 * minimum_pool_load + max(1, target_concurrency / replica_count)`.
This lets a hot adapter spread across the pool and lets small workloads use idle
GPUs. Among eligible replicas the router prefers the exact adapter version already
active or recently used. Cold placement prefers a replica whose active adapter
working set fits within `sglang_cfg.max_loras_per_batch`.

Observations are approximate. Recent completion is a cache hint, not proof of GPU
residency. Counts on different sampling workers are not synchronized. Local
reservations make bursts visible immediately within a worker; replica observations
incorporate other workers' traffic. This trades exact global load accounting for
bounded control traffic and a synchronous in-memory routing decision.

## Flash and consistency

The request still passes through the authenticated Modal Flash gateway. The router
sets `Modal-Flash-Upstream` to the discovered `host:port`, and removes competing
`Modal-Session-ID` and `Modal-Routing-Affinity-Key` headers. Logical conversation
affinity remains in the generation payload and data-parallel rank selection.

Pool membership is cached for five seconds. Active sampling workers refresh
replica observations at most once per second, coalescing concurrent refreshes.
Warm requests use the last snapshot while refresh runs in the background. Initial
discovery is bounded to three seconds; missing or ten-second-old observations
fall back to Flash. Transport failures, unavailable upstreams, and mismatched
replica identities invalidate routes temporarily. Every physical attempt releases
its local reservation, including cancellation.

Routing hints never authorize weights. The existing sidecar still resolves an
immutable model/version identity and binds that exact adapter to SGLang's request.
Existing exact/minimum version constraints and response checks remain enforced.
The sidecar reports its replica identity so the worker can detect a proxy that
ignored the requested upstream.

`GET /routing-state` reports active requests, active adapter identities, recent
adapter use, and successful adapter-registration count/time. Registration counts
measure `/load_lora_adapter` calls; they do **not** measure GPU adapter transfers
or evictions.

## Reproducing the DAPO comparison

The workload and tools live in `scripts/benchmarks/lora_routing/`. The flat
`deployment.py` recipe defines each arm; `LORA_ROUTING_MODE` selects `naive` or
`affinity`. Adapt its deployment names, Modal environment, volumes, and secret
names to your account. `deploy.py` permits the eight-GPU trainer to use any available GPU
region while the inference pool remains in the frontend's region.

Each client has a distinct rank-32 adapter and independently submits sampling,
PPO forward/backward, optimizer, and publication operations through the Tinker SDK.
One rollout batch may overlap that client's training; policy lag is bounded to one
update. Starts are staggered, with no cross-client step barriers. The fixed DAPO
rows, tokenizer revision, prompts, sampling parameters, and seeds are hashed into
each client's manifest. Dynamic group filtering is disabled to hold work fixed.

After deploying both arms, launch `supervise.py` against each frontend with
`--clients 16 --stagger-seconds 5 --hours 0.5 --config config.json`, separate output
directories, and the exact deployment app names in `--owned-apps`. Set
`TINKER_API_KEY` in the environment. The supervisor bounds runtime, detects client
failure, records GPU lifetimes, and stops its owned apps on completion or failure.
Run a two-client, two-update pilot first and check `report.py --validate-pilot`.

`client.py` stores per-group generated tokens, logprobs, text, rewards, and policy
versions, plus per-client event timestamps. `report.py` measures each step from
rollout start through updated sampling-policy readiness, including queueing and
publication. Use `--require-learning` to reject runs without mixed rewards and
nonzero gradients, and `--baseline PATH` to compare matching workloads.
`observe.py` records replica observations during the run through the same exact
Flash routing path.
