# Base-model asset preparation benchmark

The baseline reused the downloaded weights. It still repeated asset preparation on every
client creation and sample submission. Coalescing preparation removed that overhead;
in this single-frontend experiment, the lock and in-memory ready set provided the benefit
without requiring the persistent completion marker.

## Results

All 435 rollouts succeeded and returned the same 12 generated tokens. Each variant
fetched the four weight shards once: **19,306,310,872 bytes (19.3 GB)**. There were
no repeated weight-file fetches, including in the baseline.

Counts from the 48-rollout diagnostic pass:

| Variant | Preparation calls | Snapshot calls | Weight-shard fetches |
| --- | ---: | ---: | ---: |
| Without PR | 80 | 80 | 4 |
| Lock + ready set, no marker | 1 | 1 | 4 |
| Full PR, including marker | 1 | 1 | 4 |

The baseline made 32 preparations for its first batch (16 creations + 16 samples),
16 for the second batch, and 32 for new clients on the third batch. Each PR variant
made one preparation for the entire pass. All cold clients finished creation after
that initial preparation completed.

Warm measurements below used **uninstrumented code**. Each cell is the median wall
time to finish a batch of 16 clients, across three rounds:

| Variant | New clients + first rollouts | Another rollout on existing clients |
| --- | ---: | ---: |
| Without PR | 37.68 s | 21.82 s |
| Lock + ready set, no marker | 3.13 s | 2.96 s |
| Full PR, including marker | 5.21 s | 3.39 s |

The full PR improved median warm new-client batch time by **7.2×** in this run.
These are three-round observations, not a throughput or tail-latency guarantee.
The lock-only existing-client rounds were 2.96, 22.19, and 2.31 seconds. In the
22.19-second round, 15 requests finished in under 1.9 seconds; one `/asample`
request spent 21.4 seconds executing before returning. Its cause was not isolated.
All rounds, including this outlier, are retained in the [measurement data](assets/base-assets-benchmark.json).

The instrumented cold batch also included downloading, volume commit, pool deployment,
and inference startup:

| Variant | Initial snapshot | Entire first 16-client batch | All clients created after assets ready |
| --- | ---: | ---: | ---: |
| Without PR | 54.7 s | 311.7 s | 19.7 s |
| Lock + ready set, no marker | 42.9 s | 281.0 s | 17.2 s |
| Full PR, including marker | 121.7 s | 482.9 s | 15.9 s |

Cold inference startup dominated these first rollouts. The PR avoids repeated
preparation after the initial download; it does not remove inference startup time.
The marker added no observable preparation-count benefit within one frontend process.
It can skip preparation checks after process restarts, which is outside this comparison.

All experimental apps and GPU pools were stopped after measurement, and their dedicated
volumes, dictionaries, and API secrets were deleted.

## Method

Ran on 2026-09-29 against real Modal deployments in `kailash-dev`, region
`us-west`, using `Qwen/Qwen3.5-9B-Base` and 16 concurrent sampling clients
created with `create_sampling_client_async(base_model=...)` from one Tinker
service client. Each inference pool used one H200, tensor parallelism 1,
16K context, and at most one replica. Each frontend was restricted to one
container to measure the process-local coalescing guarantee. No trainers ran.

Compared three sources:

- Baseline: `d278a2f3e16624df8d883661b699bb1f9e5c8ed9`.
- Lock only: `5ef6fd4698901d574ad859c7136fc9ba0de384ad`, with only
  `prepare_model_assets` restored to its baseline body. The per-model lock
  and in-memory ready set remained enabled.
- Full PR: `5ef6fd4698901d574ad859c7136fc9ba0de384ad`, including the marker.

The cold diagnostic pass used a separate empty asset volume for each
variant. It ran three waves: 16 new sampling clients and their first
rollouts, another rollout on those clients, then 16 more new clients and
first rollouts. Instrumentation counted remote preparation requests,
`snapshot_download` calls, and Hugging Face `xet_get`/`http_get` file fetches.
The baseline would have stopped if it fetched the same large weight file
again. Instrumentation recorded synchronous Modal Dict events, so its
latency includes observation overhead.

For the warm timing pass, restored each variant's source without those
hooks, redeployed its frontend, kept its downloaded assets and inference
pool, and completed one untimed warmup request. Then ran three rounds,
each with 16 new clients plus their first rollouts, followed by another
rollout on those same clients. Reported times are medians of the three
whole-batch durations. The committed client script was used for these
measurements. The prompt was `The capital of France is`, temperature 0,
seed 42, and a maximum of 32 generated tokens.

These measurements separate actual file fetches from cache checks.
Payload sizes describe files materialized by Hugging Face, not network
wire bytes; Xet may serve chunks from caches. Cold inference startup,
image availability, and shared kernel caches can vary between runs, so
the cold wall times do not isolate the effect of this PR. The timing
comparison uses warmed inference pools and uninstrumented code.

This does not test interrupted-download recovery or simultaneous frontend
replicas. The marker's persistence across frontend restarts is a separate
property; the in-memory lock and ready set reset when a frontend restarts.

## Repeat the latency test

Deploy the desired revision using an isolated frontend, a matching inference
app, and a dedicated asset volume. Restrict the frontend and inference pool
to one container/replica as above. Set `TINKER_API_KEY` for that frontend and
install `tokenizers` alongside the repository's dependencies, then run:

```bash
python scripts/benchmark_base_sampling.py \
  --base-url "$SPINDLE_TEST_URL" \
  --model Qwen/Qwen3.5-9B-Base \
  --clients 16 --rounds 3 --warmup \
  --output sampling-timings.json
```

For a cold measurement, use a fresh asset volume and omit `--warmup`.
The script saves per-request timings, generated tokens/text, phase timings,
and partial results on failure. It does not deploy or stop apps. Stop the
isolated frontend, inference provisioner, and dynamic inference pool after
measurement. This latency script alone does not count weight downloads.
