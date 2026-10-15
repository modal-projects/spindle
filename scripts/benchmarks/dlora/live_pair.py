"""Run independent SDK RL loops on an unchanged trainer and inference pool."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import tinker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = args.root
    study = root.parent
    endpoint = json.loads((root / "endpoint.json").read_text())
    owned = json.loads((root / "owned-apps.json").read_text())
    config = json.loads((root / "measured-config.json").read_text())
    env = os.environ | {
        "MODAL_ENVIRONMENT": "lilo-deploy",
        "PYTHONPATH": "src:.",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }

    def run(arguments, name):
        with (root / (name + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, *arguments],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        # Start trainer initialization while the independent inference warmup
        # runs. This model only holds the trainer until the first measured client.
        service = tinker.ServiceClient(base_url=endpoint["base_url"])
        trainer = service.create_lora_training_client(
            config["model"],
            rank=config["rank"],
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
        )
        initialization_model = trainer.model_id
        (root / "initialization-model.json").write_text(
            json.dumps(initialization_model)
        )
        deadline = time.monotonic() + 900
        while not (root / "prepare-kv.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference preparation did not complete")
            time.sleep(5)
        for arm in ["kv", "dlora"]:
            if arm == "dlora":
                run(
                    [
                        "scripts/benchmarks/lora_routing/live_deployment.py",
                        "--run",
                        args.run,
                        "--policy",
                        "dlora",
                        "--trainer-gpu",
                        "H100",
                        "--trainer-gpus",
                        "4",
                        "--trainer-tp",
                        "4",
                        "--inference-replicas",
                        "4",
                        "--region",
                        "any",
                        "--merge-profile",
                        str(study / "merge-profile.json"),
                        "--work-rates",
                        str(study / "work-rates.json"),
                        "--root",
                        str(root),
                    ],
                    "deploy-dlora",
                )
                run(
                    [
                        "scripts/benchmarks/dlora/configure.py",
                        "--pool",
                        endpoint["pool"],
                        "--profile",
                        str(study / "merge-profile.json"),
                        "--replicas",
                        "4",
                        "--enabled",
                        "--warm-clients",
                        str(study / "pool-1/clients16.json"),
                        "--output",
                        str(root / "prepare-dlora.json"),
                    ],
                    "prepare-dlora",
                )
            run(
                [
                    "scripts/benchmarks/lora_routing/supervise.py",
                    "--root",
                    str(root / arm),
                    "--base-url",
                    endpoint["base_url"],
                    "--config",
                    str(root / "measured-config.json"),
                    "--clients",
                    "8",
                    "--stagger-seconds",
                    "5",
                    "--hours",
                    "1",
                    "--require-stable-trainer",
                    "--keep-apps",
                    *(
                        ["--initialization-model", initialization_model]
                        if arm == "kv"
                        else []
                    ),
                    "--owned-apps",
                    *owned,
                ],
                arm,
            )
        (root / "pair-complete.json").write_text(json.dumps({"time": time.time()}))
    finally:
        (root / "STOP").touch()


if __name__ == "__main__":
    main()
