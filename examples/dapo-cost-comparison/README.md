# DAPO math: matched Spindle/Tinker clients

Eight independent LoRA clients on `Qwen/Qwen3.5-9B` (not Base), 30 optimizer
updates each, started 150 seconds apart. Each client is the same standalone
`client.py`; only the endpoint and API key change between Spindle and Tinker.
The clients share an 8×H200 trainer and eight 1×H200 inference replicas on Spindle.

## Single client

Install the pinned dependencies in `requirements.txt` with Python 3.11 or 3.12.
Set the Spindle URL and API key ([server setup](https://github.com/modal-projects/spindle#shared-deployment-quick-start)).

```bash
export TINKER_BASE_URL='YOUR_SPINDLE_URL'
export TINKER_API_KEY='YOUR_KEY'
python client.py --output runs/spindle-single
```

For Tinker, use the same files and dependencies, change `TINKER_API_KEY` and unset
`TINKER_BASE_URL` to use the SDK's default Tinker endpoint:

```bash
unset TINKER_BASE_URL
export TINKER_API_KEY='YOUR_TINKER_KEY'
python client.py --output runs/tinker-single
```

Outputs must be new directories. No Modal dependency is imported by the single
client. Every client uses the same dataset order and sampling seeds;
client number changes only the output directory, not the workload.
Adapter initialization seeds are omitted on both endpoints because Miles rejects
per-model seeds. Initial adapter tensors are not guaranteed to match.

## Matched configuration

`config.json` is the shared source of truth:

- Thinking explicitly enabled through the pinned Qwen chat template.
- 16,384 maximum output tokens; 32,768 total context. Stop on model EOS/im-end.
- Temperature 1, top-p 1, top-k disabled, deterministic per-group seeds. These are
  explicit common Tinker sampling settings, not Qwen's separate OpenAI-server
  recommendations with presence penalties. No backend-specific penalties are used.
- Rank 32; attention and MLP LoRA enabled, unembedding LoRA disabled.
- Eight fixed groups of eight samples per optimizer update, with no filtering or replacement.
- Group-centered/sample-standard-deviation-normalized advantages, token-mean PPO
  with clip bounds 0.8/1.28, learning rate 1e-5, Adam 0.9/0.95, epsilon 1e-12,
  gradient clipping 1, no weight decay or reference KL penalty.
- One batch prefetched during training; at most one optimizer update of policy
  lag. Immutable named sampler snapshots prevent mid-group policy changes.
- Publish sampler weights after updates 1–29. Exactly 30 training updates, no
  hidden optimizer warmups, evaluations, or full checkpoints in the measured run.
- Eight short (128-token maximum) sampling warmup responses per client, logged
  and included in token costs. Client zero finishes this before the stagger clock
  starts, avoiding an initial shared inference cold start synchronizing the clients.

The fixture is the original 320-row numeric-answer DAPO subset, not the entire
17k dataset. Each row retains its source index from
[open-r1/DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed).
The grader conservatively compares the last boxed numeric answer. Constant-reward
groups stay in the batch with zero group-centered advantages and still pass through
forward/backward. `dynamic_group_filtering=false` fixes the workload at 64
responses per update. Variable response lengths still mean token counts and FLOPs
can differ across backends. The historical filtered run used this flag set to true.

This is a DAPO-style systems workload, not a reproduction of every component of
the original DAPO paper. It differs from the historical sweep's Base model,
plain-text rendering, 4k cap, and unnormalized importance-sampling objective.

Identical settings do **not** imply identical generations or initialization across
provider implementations. The manifest hashes the dataset, rendered prompt token
IDs and config and records SDK/tokenizer versions and stop IDs. Model weight
revisions/LoRA scaling inside Tinker are not independently verified by this runner.
Compare output-length distributions, truncation rates, reward and tokens as well
as cost; do not treat shorter, lower-quality or truncated responses as a win.

## Spindle experiment

`deployment.py` uses one 8×H200 node, eight adapter slots, and eight TP=1 sampling
replicas. `trainer_region='us'` allows broader GPU placement while the frontend
routing region remains `us-west`. Credentials use existing deployment secrets;
no key is stored in the repository.

`experiment.py` deploys the fixed TP=1/DP=8 layout and runs `supervise.py`.
There is no synthetic profiling stage in the launch path. TP=1/DP=8 is a candidate
chosen to favor data-parallel throughput; it has not been proven fastest.
`profile.py` remains an optional standalone diagnostic, separate from the run.

The supervisor launches independent copies of `client.py`, with no synchronization
barrier after startup. Actual process launch times and per-client readiness are
recorded. Fixed offsets do not guarantee clients remain separated as generation
lengths vary. The training client uses only ordinary Tinker SDK APIs and imports
no Spindle or Modal modules.

On this host:

```bash
cd ~/spindle-dapo-comparison
source ~/spindle/.env
export MODAL_ENVIRONMENT=lilo-deploy
/tmp/swe-rl-venv/bin/python examples/dapo-cost-comparison/experiment.py \
  --root scripts/results/NEW_RUN \
  --deploy-python ~/spindle/.venv/bin/python
```

Training has a 12-hour supervisor limit. The supervisor stops only its explicitly
owned apps on failure/completion.
Do not run two pipelines simultaneously with these fixed deployment names.
Create `STOP_NOW` in the training results folder to request training shutdown.

## Cost and interleaving evidence

Each output directory contains its manifest, an event log, and gzipped raw groups
with exact prompt/generated tokens, behavior logprobs, text, reward, and stop reason.
All generated tokens count, including discarded constant-reward groups and sampler
warmup. Retained/generated token counts are reported separately. Client events mark
forward/backward + optimizer request intervals; these include queueing and are not
physical GPU occupancy measurements.

The supervisor records provider container lifetimes, including the initial and
staggered idle periods. GPU-seconds alone are not a complete invoice: CPU, memory,
storage and provider discounts can change total cost. It does not assume Tinker
prefix-cache hit rates or call a token-price estimate an actual Tinker run.

```bash
python report.py PATH_TO_SPINDLE_TRAINING --compare PATH_TO_TINKER_SINGLE
# Optionally supply actual run-attributed billing totals:
python report.py PATH_TO_SPINDLE_TRAINING --compare PATH_TO_TINKER_SINGLE \
  --billed-usd YOUR_SPINDLE_TOTAL --comparison-billed-usd YOUR_TINKER_TOTAL
```

Divide the complete eight-client Spindle cost by eight for per-client cost and
compare against one complete 30-update Tinker run. Also show total generated tokens,
reward, truncation, and per-client batch latency. Cache/billing details and failed
request retries should be reconciled with the provider usage export. A completed
run and complete lifetime coverage are prerequisites for a final cost conclusion.

API references: [sampling parameters](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/samplingparams/),
[PPO clipping](https://tinker-docs.thinkingmachines.ai/tinker/losses/ppo/).

### Trainer activity timeline

Collect the trainer app's logs from before the run started, then render the
backend activity and each client's outstanding training requests:

```bash
python -m modal app logs spindle-trainer-dapo-instruct-tp1-dp8-20260930 \
  --env lilo-deploy --since 12h > trainer.log
python examples/dapo-cost-comparison/trainer_activity.py RUN_DIRECTORY \
  --trainer-log trainer.log --through "$(date +%s)"
```

The plotting script needs `matplotlib` and `numpy`; these are analysis-only
requirements. It writes `activity/trainer-activity.{png,pdf,json}`. Rerun with a
fresh log snapshot to refresh the chart. `--through` is the end of log collection;
log coverage must include the entire measurement window.

Solid bars show completed backend forward/backward, optimizer, and snapshot
capture calls. The black line is centered 60-second call occupancy, counting the
union of intervals. Hatched backend bars show an unfinished rank-zero call and
are excluded from the completed-call percentage. Host call time includes backend
synchronization and compilation; it is not hardware GPU utilization. Client lanes
show submission-to-response time **including queueing**, not attributed GPU time.
Current backend logs identify only the first model in a shared batch, so they do
not support exact execution attribution to every participating client. The chart
starts at the first client's readiness and includes stagger and rollout waits.
This analysis does not change the portable training client or deployment.

The current Spindle response format requires SDK 0.24.1; Tinker currently requires
a newer SDK (0.30.4 used in the comparison). This version difference must be
reported until Spindle supports the new protobuf sampling responses.


## Infrastructure optimization trials

`deployment_optimized.py` and `experiment_optimized.py` run short infrastructure
trials with the same `config.json` workload. `--updates 3` changes only the trial
length; the eventual full run uses `--updates 30 --stagger-seconds 150`.
Short trials use a 20-second stagger to exercise concurrent load within three
updates, so their startup/queueing behavior is not directly comparable with the
150-second full-run stagger. Each new trial records its client, recipe, and Git
diff alongside its configuration.

The candidate recipe controls frontend/sampling-worker CPU, inference tensor
parallelism and replica count, and trainer compilation through flat settings.
Training hyperparameters, prompts, rewards, group counts, sampling parameters,
and the one-update policy-lag limit remain unchanged. The portable client queues
its immutable snapshot request immediately after the optimizer request, before
waiting for their results; server sequence ordering still places capture after
the optimizer. `publish` measures the remaining wait plus sampling-client
creation, and `publish_enqueued` records when the snapshot was queued.

Server transport supports zstd-compressed protobuf training requests and gzip
responses. These are negotiated with the standard SDK and do not change the
training payload. Report pilot startup/compilation separately from steady-state
step latency, but include both in resource cost. Keep pilot costs separate from
full-run cost and label GPU-rate projections as estimates, not invoices.
