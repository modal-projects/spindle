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
It does not stop the older H200 experiment. Results are pending; this document
is an evaluation specification, not a performance claim.
