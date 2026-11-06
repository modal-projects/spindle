"""Separate rollout workers and sandbox ownership for the Miles experiment."""

import modal

from .workers import image as base_image
from .workers import measure_capacity as base_measure_capacity
from .workers import sample_trajectory as base_sample_trajectory

app = modal.App("spindle-mimo-miles-rollouts")
image = base_image


@app.function(
    image=image,
    cpu=2,
    memory=16384,
    timeout=18000,
    max_containers=512,
    secrets=[modal.Secret.from_name("spindle-proxy")],
    env={"SPINDLE_MIMO_SANDBOX_APP": "spindle-mimo-miles-sandboxes"},
)
def sample_trajectory(task, gateway, group, model_id, version, seed):
    return base_sample_trajectory.local(task, gateway, group, model_id, version, seed)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=3600,
    secrets=[modal.Secret.from_name("spindle-proxy")],
)
async def measure_capacity(gateway, model_id, version):
    return await base_measure_capacity.local(gateway, model_id, version)
