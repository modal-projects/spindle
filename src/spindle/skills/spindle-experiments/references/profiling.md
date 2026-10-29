# Profiling a step

Miles LoRA trainer only; off by default.

## Capture

- `SPINDLE_TORCH_PROFILE_STEP=2 uv run spindle deploy <config>`: traces the third optimizer step (counted from 0; use 1 or 2 so compile work is out). Unset to disable.
- `SPINDLE_TORCH_PROFILE_RANKS`: default `0` (rank 0); e.g. `0,4` or `all`. ~250 MB per trace.
- `SPINDLE_TORCH_PROFILE_DIR`: default `/checkpoints/torch-profile/<definition id>`.

Captures one optimizer step (first `forward_backward` of the step to before the next), phases labelled `spindle/forward_backward`, `spindle/optim_step`, `spindle/forward_only`, `spindle/export_slot_peft`, plus a CPU-only `controller.trace.json.gz` for post-step publish. `rank<N>.key_averages.txt` is a no-UI top-ops table.

## Download and view

```bash
uv run modal volume get spindle-checkpoints /torch-profile/<definition id>/rank0.trace.json.gz .
```

Open at https://ui.perfetto.dev (loads `.json.gz` directly). Search `spindle/` to jump between phases. Look for idle GPU streams inside `spindle/forward_backward` (CPU/comms stalls), long `ncclKernel_*`, and time outside labelled phases.

## DuckDB queries

The `.json.gz` is one JSON object; raise the object-size cap before unnesting:

```sql
CREATE VIEW ev AS
SELECT unnest(traceEvents, recursive := false) AS e
FROM read_json('rank0.trace.json.gz', format='unstructured',
               maximum_object_size := 536870912);
```

Duration of each `spindle/*` phase:

```sql
SELECT e->>'name' AS phase, sum((e->>'dur')::DOUBLE) / 1e6 AS dur_s
FROM ev
WHERE e->>'cat' = 'user_annotation' AND e->>'name' LIKE 'spindle/%'
GROUP BY 1 ORDER BY 2 DESC;
```

Top GPU kernels inside one phase's window (`kernel`/`gpu_memcpy`/`gpu_memset` are the GPU-stream categories; `gpu_user_annotation` marks phases on GPU rows):

```sql
WITH w AS (
  SELECT (e->>'ts')::DOUBLE ts, (e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE ts_end
  FROM ev
  WHERE e->>'name' = 'spindle/forward_backward' AND e->>'cat' = 'user_annotation'
  LIMIT 1
)
SELECT e->>'name' AS kernel, sum((e->>'dur')::DOUBLE) / 1e3 AS ms
FROM ev, w
WHERE e->>'cat' = 'kernel'
  AND (e->>'ts')::DOUBLE >= w.ts AND (e->>'ts')::DOUBLE < w.ts_end
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

GPU busy fraction of the phase window (sums overlapping streams, so >1 means parallel streams, not >100% of one GPU):

```sql
WITH w AS (
  SELECT (e->>'ts')::DOUBLE ts, (e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE ts_end
  FROM ev
  WHERE e->>'name' = 'spindle/forward_backward' AND e->>'cat' = 'user_annotation'
  LIMIT 1
), k AS (
  SELECT greatest((e->>'ts')::DOUBLE, w.ts) s,
         least((e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE, w.ts_end) f,
         w.ts_end - w.ts AS win
  FROM ev, w
  WHERE e->>'cat' IN ('kernel', 'gpu_memcpy', 'gpu_memset')
    AND (e->>'ts')::DOUBLE < w.ts_end
    AND (e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE > w.ts
)
SELECT coalesce(sum(greatest(f - s, 0)) / max(win), 0) AS busy_frac FROM k;
```

NCCL share of kernel time:

```sql
SELECT sum((e->>'dur')::DOUBLE) FILTER (e->>'name' LIKE 'nccl%')
     / nullif(sum((e->>'dur')::DOUBLE), 0) AS nccl_frac
FROM ev WHERE e->>'cat' = 'kernel';
```
