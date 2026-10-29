# Two sampler pools, one trainer

`groups.py` is one Qwen3.5-4B trainer and two inference deployments: H100 BF16,
scheduled in any region, and H200 FP8 on the platform region. Both pools should
be attached to the trainer's model id:

```python
pool = FFTPoolSpec(deployment.definition_id, training.model_id, True, 0)
app, server = build_rollout_app(deployment, pool)
```

The trainer publishes BF16. The FP8 pool requantizes. `trainer_region = ()`
and `inference_region = ()` leave compute unpinned; a name or a tuple of names
pins it. `trainer_rdma = True` requests RDMA for a single node as well as a
multi-node cluster.

`train.py` starts the rollout when the first URL answers and cancels the other
probe when the update finishes. A sampler that becomes ready during the update
is still recorded.

```bash
uv run python examples/heterogeneous-samplers/train.py \
  --url h100-bf16=https://... --url h200-fp8=https://...
```
