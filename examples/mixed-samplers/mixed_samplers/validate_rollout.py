"""One real coding trajectory through the experiment's sampler and sandbox path."""

import argparse
import gzip
import json
import pickle
from contextlib import ExitStack
from pathlib import Path

import modal

from spindle.deployments import DeploymentConfig
from spindle.providers.modal.deployment_apps import build_rollout_app
from spindle.providers.modal.fft_pool import FFTPoolSpec

from .config import GROUPS, SEED, sampler_config
from .data import load_tasks
from .workers import app as worker_app
from .workers import sample_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", choices=[g["name"] for g in GROUPS], default="h200-fp8"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    group = next(g for g in GROUPS if g["name"] == args.group)
    deployment = DeploymentConfig.create(sampler_config(group))
    app, server = build_rollout_app(
        deployment, FFTPoolSpec.base(deployment.definition_id)
    )
    task = load_tasks()[args.task]
    with modal.enable_output(), ExitStack() as stack:
        stack.enter_context(app.run(environment_name="kailash-dev"))
        stack.enter_context(worker_app.run(environment_name="kailash-dev"))
        compressed = sample_trajectory.remote(
            task, server.get_url(), group["name"], None, 0, SEED
        )
        args.output.write_bytes(compressed)
    row = pickle.loads(gzip.decompress(compressed))
    print(
        json.dumps(
            {
                key: row[key]
                for key in (
                    "instance_id",
                    "group",
                    "reward",
                    "format_error",
                    "stop_reason",
                    "turns",
                    "generated_tokens",
                    "verifier",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
