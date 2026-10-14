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

After the Domains and Work turnover arms, a CPU placement audit found that
registering 16 new versions changed existing assignments 33 times with the
even-count map (using the measured model and replica identities). The original
slot-budget map had no such moves in this specific audit, but assigned between
one and five initial adapters per replica. Add job-identity static and overflow
variants to the supplemental screen, before any supplemental GPU result. These
balance primary jobs and preserve their homes across publications; cache reuse,
residency, reservations, and executed weights remain exact-version scoped. This
audit is a constructed placement test, not a measured count of GPU migrations.

The long-output bounded-KV arm failed with an HTTP/2 connection-state error.
An auxiliary preparation request propagated the error, and incomplete sibling
cancellation then closed a shared transport while other requests were active.
Preserve the failed arm and original manifest. Separate preparation traffic
from generation and drain nested cancellations before closing clients. Rerun
all seven long-output policies with that same transport; later supplemental
arms use it as well. This transport change is recorded in the workload digest.
The earlier three completed long-output arms remain exploratory observations.

Separation alone did not fix the HTTP/2 failure. The next validation gives each
generation group its own authenticated HTTP/2 transport, matching production
sample_task, and uses HTTP/1.1 for the low-concurrency auxiliary preparation.
Protocol diagnostics record GOAWAY codes and stream IDs without headers or
payloads. Both failed attempts remain preserved; neither supports a performance
claim. All subsequent long-output arms use the same recorded transport settings.

The 64-version screen also motivates a different work estimate. A transfer-only
adapter cost does not represent native admission loss at the eight-adapter batch
limit. For remaining decode budgets d_r, sequence capacity C, and adapter
capacity S, estimate decode iterations by the maximum of: max_r d_r,
ceil(sum_r d_r / C), and ceil(sum_a max_{r using a} d_r / S). Multiply by the
measured full-batch iteration time C / decode_tokens_per_second, preserving the
common prefill and loading terms. These are necessary iteration counts for known
lengths, not a latency guarantee; variable batch time, nonpreemptive admission,
and natural-EOS lengths introduce error. Add per-adapter remaining maxima to the
CPU observation, and require that schema before using the new estimate.

Screen this candidate on a second, independently monitored eight-H200 pool
against KV, static exact-version placement, and job-identity overflow. Reuse
the original calibration without fitting to these new outcomes. All policies
on this pool use the same expanded observation schema. Its six-hour deadline
and the first pool's deadline are independent; the combined task allocation
ceiling is 32 GPUs while the live validation is active. Selection still waits
for both the simple controls and this candidate's screening evidence.

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

The per-task HTTP/2 preparation attempt completed the two-choice arm but timed
out in static placement; another pool completed job overflow and timed out in
native KV. These later failures are read timeouts, not the original HTTP/2 state
error. Engine logs show a cohort request being aborted with active generation
and queues present in the static arm. The helper still had a hard-coded
240-second timeout while measured generation allowed 600 seconds. Use the same
configured request timeout for both, and record it explicitly. This does not
establish the cause of every earlier failure. Before further helper comparisons,
run the production admission path without cohort preparation.

The capacity-score candidate is rejected after its first serving screen:
2,386 output tokens/s versus 3,824 for static placement, with 1,326 versus 168
GPU adapter loads. Job-identity overflow reaches 3,629 tokens/s on that pool,
5.1% below static. Before selection, add two conservative count-based controls:
retain the job home within one effective engine batch of the minimum request
count; or spill only when the home has observed pressure at the engine's running
capacity and another replica has spare capacity after local reservations.
Neither is treated as a selected policy. These controls test whether aggressive
overflow itself explains the lost locality without fitting additional costs.

## Live RL

During screening, run one KV-affinity baseline validation to establish actual
output lengths, filtering cost, nonzero gradients, and independent progress.
This two-update validation uses 16 GPUs under a 150-minute watchdog, overlapping
the separately monitored eight-GPU screen (24 GPUs combined). It does not select
the winning routing policy. Give each client a fixed 20-problem offset into the
320-problem DAPO dataset, identical across compared arms, to avoid 16 copies of
one problem stream. Use dynamic group filtering with a 64-candidate-group cap.
The initial local SDK credential did not match the deployment secret; that run
stopped before model creation. Its replacement authenticated successfully, but
the eight-H200 trainer remained queued with zero runners for approximately ten
minutes. Both allocations were stopped and audited. Continue validation with
eight H100 trainer GPUs, TP=4/DP=2 and a 16,384-token microbatch budget, retaining
eight H200 inference replicas. The client workload and original absolute pilot
shutdown deadline remain unchanged. Use the deployment's existing API secret
only in the child environment; do not print or persist its value.

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

For paired live runs, save each client's initial training checkpoint in the
first arm and restore that client's weights before later arms. Miles rejects
per-model initialization seeds; a saved state avoids silently comparing different
initial adapter matrices. Perform save/load before sampling warmup and retain
the source checkpoint paths in the event record. Initialization is reported
separately from rollout-to-publication step time.

The first live pilot reached its two-hour client deadline with 31 of 32 updates
published and 15 of 16 clients complete. All 31 optimizer gradients were finite
and nonzero. The last client continued sampling, but a one-group refill and a
sequence of constant-reward groups produced a long filtering tail. Preserve
the incomplete run; do not extrapolate its aggregate training throughput. Its
GPU apps were stopped at the deadline. For subsequent paired runs, permit a
fixed minimum refill width, identical for every policy. Retain exactly the
requested training batch; count nonmixed filtered groups separately from
surplus mixed groups, and include all completed generation in resource metrics.

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
