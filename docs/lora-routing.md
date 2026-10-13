# LoRA-aware rollout routing

LoRA sampling workers choose replicas locally. Each worker maintains a periodically
refreshed snapshot of the pool and counts its own in-flight generations. There is
no shared scheduler, distributed lock, or routing RPC on a warm request's path.

See the [measured results](lora-routing-results.md) for the completed 16-adapter,
27B-model training comparison, matched inference replay, and their limitations.

`inference_lora_routing = "affinity"` enables routing; `"naive"` keeps the existing
adapter-independent Flash behavior for comparisons. Base-model sampling and FFT
routing keep their existing paths.

When upgrading an existing deployment, refresh its inference sidecars with
`spindle deploy config.py --refresh-inference CONFIG_NAME`. Pools without the
`/routing-state` endpoint continue using Flash's ordinary routing until refreshed.

## Placement and load

New prefixes start with rendezvous hashing of their version-scoped affinity key,
so a model's many independent problems spread across replicas when capacity is
available. If that preferred replica's active adapter working set is full and
does not contain the requested adapter, cold placement prefers an eligible
replica already serving that adapter, then one with available adapter capacity.
These fallback candidates use deterministic model rendezvous order. Adapter
concentration is a response to working-set pressure, not the default for every
new prefix.

Each exact adapter-version/affinity-key pair keeps its first destination in a
bounded local map. A later residency change does not remap it. Sidecars advertise
bounded, version-scoped recent prefix keys so another sampling frontend can
recover locality. If the home is overloaded, the router first looks for another
eligible replica with that prefix, then uses adapter placement. Temporary overflow
does not rewrite the original home. Failed/departed homes can be replaced.

Estimated load adds new local reservations to reported outstanding requests,
without double-counting reservations already present at observation time. A
replica is eligible below `minimum_observed_load + target_concurrency`.
The configured target is an explicit budget of excess outstanding requests that
we are willing to accept for locality. At low load it bounds concentration; at
high load it preserves a small prefix group instead of spreading each sample
across hosts just above the pool mean. A hot home spills once it consumes this
budget. This is a request-count heuristic, not a calibrated prefill/decode cost
model or a proof that one target is the optimal budget for every workload.


These are approximate observations, not a distributed admission guarantee.
Independent frontends can overshoot the bound between refreshes. Recent prefix
use does not prove KV residency, and recent adapter use does not prove GPU
residency. The engine still owns cache validation. A prefix may be worth keeping
even when its adapter must be reloaded: changing adapter warmth alone never
moves an established prefix. Stale observations fall back to native Flash.

## Flash and consistency

The request still passes through the authenticated Modal Flash gateway. The router
sets `Modal-Flash-Upstream` to the discovered `host:port`, and removes competing
`Modal-Session-ID` and `Modal-Routing-Affinity-Key` headers. Logical conversation
affinity remains in the generation payload and data-parallel rank selection.

Pool membership is cached for five seconds. Active sampling workers refresh
replica observations at most once per second, coalescing concurrent refreshes.
A persistent HTTP client reuses observation connections. While generations remain
in flight, a per-router background task keeps this same coalesced refresh cadence
active; it exits when the router has no local reservations. This avoids using an
old burst snapshot for the next independently arriving client. Prefix maps are bounded
to 8,192 entries per router and 4,096 per sidecar; sidecar hints expire after 120s.
Fresh requests use the last snapshot while refresh runs in the background.
Initial discovery and bursts after ten seconds of idle observations share one
refresh, bounded to three seconds. Missing or still-stale observations then fall
back to Flash. This avoids sending an entire post-idle batch through naive routing
while its refresh is pending. Transport failures, unavailable upstreams, and mismatched
replica identities invalidate routes temporarily. Every physical attempt releases
its local reservation, including cancellation.

Routing hints never authorize weights. The existing sidecar still resolves an
immutable model/version identity and binds that exact adapter to SGLang's request.
Existing exact/minimum version constraints and response checks remain enforced.
The sidecar reports its replica identity so the worker can detect a proxy that
ignored the requested upstream.

`GET /routing-state` reports active requests, active adapter identities, recent
adapter use, recent version-scoped prefix keys, and successful adapter-registration count/time. Registration counts
measure `/load_lora_adapter` calls; they do **not** measure GPU adapter transfers
or evictions.

## Reproducing the DAPO comparison

The workload and tools live in `scripts/benchmarks/lora_routing/`. The flat
`deployment.py` recipe defines each arm; `LORA_ROUTING_MODE` selects `naive` or
`affinity`. Adapt its deployment names, Modal environment, volumes, and secret
names to your account. `deploy.py` permits the eight-GPU trainer to use any
available GPU region while the inference pool remains in the frontend's region.

Each client has a distinct rank-32 adapter and independently submits sampling,
PPO forward/backward, optimizer, and publication operations through the Tinker SDK.
One rollout batch may overlap that client's training; policy lag is bounded to one
update. Starts are staggered, with no cross-client step barriers. The fixed DAPO
rows, tokenizer revision, prompts, sampling parameters, and seeds are hashed into
each client's manifest. Dynamic group filtering is disabled to hold work fixed.

After deploying both arms, launch `supervise.py` against each frontend with
`--clients 16 --stagger-seconds 5 --hours 0.75 --config config.json`, separate output
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

For example, after collecting both runs and the replica observations:

```bash
python scripts/benchmarks/lora_routing/report.py RESULTS/affinity \
  --validate-pilot --require-learning --baseline RESULTS/naive \
  --observations RESULTS/observations.jsonl --pool-app YOUR_AFFINITY_POOL \
  --output RESULTS/affinity-report.json
```

`replay.py` supplements the training comparison with an inference-only replay of
the same saved adapters and prompts on two fresh, identical pools. It loads two
published versions from a completed 16-client run, uses eight independent router
caches, and submits four waves of 1,024 sequences, capped at 512 tokens. It checks
exact versions and finite aligned logprobs and stops both supplied pools in a
`finally` block. The pools must share the training run's bulletin/checkpoint
volumes. Pass `--pools NAIVE_POOL AFFINITY_POOL --output RESULTS/replay.json` and
the completed training output directory as the positional argument. This replay
has fixed wave barriers and must not be presented as training throughput.
Summarize its output with `report_replay.py RAW_REPLAY.json --output SUMMARY.json
--groups-csv GROUPS.csv`.
