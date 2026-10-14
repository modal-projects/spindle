# Multi-LoRA routing study

`paper.pdf` is the report; `paper.tex` and its included sections are the source.
The final report distinguishes exploratory HTTP/1 runs, transport-affected runs,
and controlled HTTP/2 comparisons. It does not treat saved-checkpoint serving
replays as new optimizer training runs.

The implementation is experimental and leaves the production routing default
unchanged. `DomainRouter` extends estimated-work placement with demand-sized
adapter domains. Cohort prefix preparation is a benchmark helper, not an enabled
production feature.

## Rebuild the analysis

From the repository root, with the saved raw experiment directory available:

```bash
MPLCONFIGDIR=/tmp/spindle-mpl .venv/bin/python scripts/benchmarks/lora_routing/paper_results.py \
  --root /home/ec2-user/lilo/analysis/pool-paper-20261014 \
  --output docs/paper/figures
```

The analysis validates completed arms, emits per-trial JSON/CSV metrics, produces
vector PDF and PNG figures, and generates the quantitative LaTeX tables. Failed
attempts remain in successful-arm latency and throughput measurements. An
incomplete arm is not silently treated as a completed trial.

Compile from this directory with Tectonic 0.17:

```bash
tectonic --keep-logs paper.tex
```

The raw directory is not committed because it contains large per-request traces
and private checkpoint provenance. Access to those checkpoints, the pinned
SGLang fork, and the Modal environment is required to repeat GPU experiments.
The committed metrics and figures are sufficient to inspect reported aggregates.
Each raw arm stores separate workload and implementation hashes.
`figures/plot-values.json.gz` contains the rolling curves and sorted request
latencies used for the time-series and CDF plots, without prompts, generated
text, or checkpoint paths. Empty completion windows are represented by JSON null.
`paired-comparisons.json` records per-trial changes and per-client rollout
speedups; it does not treat individual requests as independent trials.

## CPU measurements and counterexample

```bash
.venv/bin/python scripts/benchmarks/lora_routing/router_cpu_benchmark.py --output /tmp/router-cpu.json
.venv/bin/python scripts/benchmarks/lora_routing/router_map_benchmark.py --output /tmp/router-map-cpu.json
.venv/bin/python scripts/benchmarks/lora_routing/router_cpu_benchmark.py \
  --correlated-burst --output /tmp/stale-view-counterexample.json
```

The last command executes eight independent routers with the same empty pool
observations. It measures assignments only, not GPU performance. It exposes the
absence of cross-frontend admission control for a correlated same-adapter burst.

## GPU experiment records

`paper_suite.py --phase http2` launches the fixed comparison sequence against an
already deployed pool. It allocates no additional GPUs. Do not run it without
the independent `cache_monitor.py` shutdown process and a pool manifest scoped
to the experiment. The saved suite manifests contain exact executed commands,
policy order, start/end timestamps, and status. Workloads and transport changes
are described in the report and `../pool-routing-research-plan.md`.

The study uses eight fixed H200 replicas, below the requested 48-GPU ceiling.
The resource audit records 36.71 GPU-hours from task lifetimes, a peak of eight
GPUs, and zero active GPUs after shutdown. The controlled HTTP/2 set contains
21 completed arms and 69,568 logical requests without physical retries.

## Validation

```bash
.venv/bin/pytest -q tests/inference/test_routing_work.py \
  tests/inference/test_routing_domains.py tests/inference/test_lora_routing.py \
  tests/inference/test_sampling.py tests/inference/test_cohort_prefill.py \
  tests/providers/test_lora_routing.py tests/providers/test_deployment_apps.py \
  tests/providers/test_modal_app.py
```

The study's validation records 126 passing tests plus live greedy preparation
controls on every replica. Request traces check exact adapter versions, fixed
output lengths, finite aligned log probabilities, and empty final queues.
