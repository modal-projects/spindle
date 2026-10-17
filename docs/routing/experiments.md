# Matched RL comparison

Compare native Spindle default routing with `joint`, using the same pinned
27B engine and split-K optimization in both arms. The source baseline is
Spindle main `6cdef38131c435b088c029b2fb51e8f56c885525`. Do not attribute the
kernel gain from the older four-arm study to this router.

Each arm has eight independent rank-32 LoRA clients, starting 300 seconds apart,
ten updates, 32 groups of eight trajectories per update, and no group filtering.
The existing SDK loop prefetches one rollout during training, publishes updated
weights, and checkpoints optimizer state after every update. Arms restore the
same eight initial checkpoints. There are no cross-client update barriers.

Trainer: 8 H200, TP4/DP2. Inference: 1–4 H200, TP1, maximum 32 running and eight
queued requests per replica, Flash target concurrency 16, 16K context. One pair
uses at most 24 GPUs; the existing H200 pair remains separate. Audit both roots
against the 48-GPU total limit. Run DAPO and Codeforces pairs successively.

- **DAPO math:** unchanged dataset/prompts and reward, natural EOS, 8,192 maximum
  generation tokens. This is the decode-heavy control.
- **Codeforces revision:** the example's 123 validated training problems, with
  its 16 held-out problems excluded. Generate a solution (thinking enabled,
  4,096-token limit), execute it in the existing isolated sandbox judge, report
  actual passed/total counts, then generate a revision (1,024-token limit).
  Every branch receives feedback and revises, whether initially correct or not.
  Each branch retains its exact sampled prefix; there is no synthetic padding.
  Both turns use the same published adapter. Final codegolf reward trains both
  turns; problem/history/feedback tokens have zero loss weight. The judge never
  exposes hidden test outputs in the feedback. Record judge time separately.

Codeforces is a multi-turn extension of the example's judge/reward, run through
our established multi-client LoRA loop; the original example was single-turn
full-model training. Measure actual prompt/output lengths and prefill latency
before calling this experiment prefill-bound. A 16K cap does not establish that.

Persist per-client rollout, training/RPC wait, publication and checkpoint times;
per-turn lengths, cached tokens, rewards and behavior versions; replica running
and queued counts, completed output tokens, LoRA loads/evictions/hits, adapter
batch composition, routing mode and 503 counts. Engine latency excludes
pre-admission waits; SDK group/trajectory latency includes them. Report matched
steady windows separately from complete-job makespan, startup and stagger.

The Rust candidate changes admission waiting as well as destination selection.
A default-versus-joint result alone cannot distinguish those mechanisms. The
`least_load` policy provides the same admission/grouping machinery without KV
or LoRA scoring for a follow-up ablation if the candidate improves.

Entry points:
- `scripts/benchmarks/rl_rust/deploy.py --root ROOT --policy default|joint`
- `scripts/benchmarks/rl_rust/run.py --root ROOT --external-root EXISTING_ROOT`

The latter calls the existing recoverable supervisor, adds combined GPU
accounting and the existing trainer guard, and cleans up only its own apps.
It does not stop the older H200 experiment.

## Partial DAPO results, October 17

The first attempt committed 25 default and 30 joint client updates before a
resource-monitor timeout stopped both arms. Codeforces had not started.
The following compares the same 02:20–04:50 UTC interval, with four H200
inference replicas per arm and identical split-K engines:

| Metric | Default | Joint |
| --- | ---: | ---: |
| Completed output tokens/s | 4,650 | 4,744 |
| Token-weighted prefix cache hit rate | 40.4% | 41.7% |
| Mean accepted-request latency, seconds | 220.6 | 183.2 |
| Mean completed SDK group latency, seconds | 2,198 | 1,891 |
| HTTP 503 attempts | 1,536,161 | 80 |
| GPU LoRA loads / evictions | 83 / 80 | 127 / 123 |
| Mean running requests across four replicas | 124.9 | 127.8 |
| Mean engine-queued requests across four replicas | 30.0 | 4.0 |

Throughput increased 2.0%. Five consecutive 30-minute windows ranged from
−0.1% to +5.4%; these are subdivisions of one run, not independent trials.
The clearest change is admission behavior: fewer rejected attempts and less
engine queueing. Accepted-request latency excludes waiting in the router.
Completed group latency includes that wait but excludes timed-out groups;
three sampling timeouts occurred in the joint arm. Neither latency measure
establishes an unconditional end-to-end improvement. Increased adapter loads
also do not support a claim of reduced LoRA thrashing on this workload.

The monitor repeatedly queried 785 already-finished tasks among 860 records.
Skipping immutable finished records and bounding concurrent queries reduced
the same history audit to 0.25 seconds. Recovery preserves committed optimizer
checkpoints and the original client inputs. Analyze resumed intervals separately.
Raw counters, interval bounds and the analysis script are in the experiment's
`dapo/partial-comparison-20261017.json` and `dapo/analyze-partial.py` artifacts.
