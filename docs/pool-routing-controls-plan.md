# Stronger controls and live RL, October 14

This follow-up is specified before collecting its new GPU results. The first
paper's two-trial comparisons and all regressions remain part of the record.
The objective is a routing policy that improves on credible simple controls,
not merely on native KV affinity in the 64-version workload.

## Controls and decision process

All new controls share observations, exact-version cache identity, local
reservations, chunk sizes, failure handling, and Flash transport. Compare:

- Native Flash KV affinity.
- Least observed request count, including unobserved local reservations.
- Static adapter placement using the same slot-balanced primary map as Domains.
  This deliberately strong control isolates the value of dynamic routing from
  the value of simply packing adapters into the aggregate GPU slot budget.
- Two-choice adapter placement: the balanced primary and a deterministic
  secondary, choosing the less loaded replica by request count.
- Adapter placement with overflow: retain the balanced primary until its load
  exceeds 1.25 times pool mean plus one placement chunk, then use least load.
- Even-count static placement and its overflow variant. The original primary
  map balances slot budgets, so a catalog smaller than total GPU slot capacity
  can still have unequal adapter counts. These stronger controls assign the
  next adapter to a replica with the fewest assigned primaries, using rendezvous
  order to break ties. They isolate that mapping choice from dynamic scheduling.
- Bounded cache affinity: maximize actual reusable prefix among replicas with
  request count at most 1.25 times the mean plus one incoming placement chunk;
  break ties by request count. This rule ignores adapter residency.
- Unrestricted measured-work routing.
- The previously reported Domains policy, without changing its algorithm.

The adapter-overflow control was added while the first KV arm was running,
before any screening results were available. It will run as a supplemental
job because the first seven-policy command was already submitted. The prior
LoRA/affinity heuristic is also retained as an implementation baseline in the
single-sequence serving comparison, with its existing behavior unchanged.
The even-count controls were added after the first static serving result and
before any turnover screening result; they also run as supplemental jobs.

First screen every policy on the original 64-version serving shape and a
fixed-work 1,024-output turnover shape. The latter uses three full rounds, so
every client reaches a new adapter version and policy-dependent admission
counts cannot conceal a regression. Screening runs select candidates, not
confirmatory evidence. Preserve unsuccessful policies and results.

Revise the design if these controls explain its gains or reveal an unresolved
failure. Freeze the selected implementation before five further trials with
new prompt seeds and counterbalanced policy order. Compare with native KV and
the strongest applicable simple control on adapter pressure, warm-prefix
reuse/version turnover, and dependent multi-turn generation. Include longer
outputs and natural-prompt cases; do not select a winner solely by an aggregate
across workloads that hides a large regression in one family. Record seeds,
commands, source hashes, request counts, and completion of every scheduled arm.
Held-out trial seeds also shuffle which saved adapter receives each client's
start offset and demand profile, and salt the affinity identities consistently
through warmup and replay. Merely changing token contents would leave the same
affinity-to-replica collisions in every trial. Screening retains the original
client order and affinity identities for direct comparison with the first study.

For comparisons with cohort preparation, give it to every policy and include
its time and input work. Also evaluate the selected path without preparation
before interpreting the benchmark helper as a deployable improvement.

## Live RL

Once serving comparisons support a candidate, run actual independent DAPO
clients through the SDK and command scheduling path. Use real forward/backward,
optimizer updates, and adapter publication. Preserve per-client prefetch/update
overlap and staggered starts; do not introduce cross-client step barriers.

The prior routing experiment's 1,536-token cap and 350-word prompt instruction
are too short for this validation. The repository's full DAPO recipe permits
20,480 output tokens. Start live validation with natural math prompts, thinking
enabled, an 8,192-token output budget, and 16,384 context capacity on the 27B
model. Record actual output-length distributions and truncation rates. Use
16 clients with at least four groups of eight samples per update (512 potential
sequences), then increase groups if observed pressure is insufficient. A short
two-update validation must demonstrate independent progress before longer runs.
Run repeated paired comparisons against the strongest deployable baseline.

Report per-client rollout-start-to-published-policy time, rollout, training,
publication, generated and trained tokens, rewards, optimizer success, policy
lag, physical attempts, and resource use. Do not infer convergence from a short
throughput experiment. If training or publication is the bottleneck, show it;
do not claim inference speedup as training speedup.

## Allocation and reporting

The first serving allocation is eight H200s with a six-hour absolute shutdown
deadline (at most 48 GPU-hours); stop it sooner when experiments finish or no
useful GPU work remains. Live RL will use separately bounded allocations with
at most 32 GPUs concurrently, below the user's 48-GPU ceiling. Do not overlap
unaccounted experiment pools. Preserve allocation and zero-GPU shutdown audits.

Regenerate the paper with all controls, individual-trial results and variability,
the selected algorithm and its reasons, real end-to-end RL measurements, and
the earlier negative evidence. If a simpler control wins, state that result
and simplify the implementation rather than relabeling it as a novel advance.
