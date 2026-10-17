"""Launch all four independent-client experiments concurrently and fail visibly."""

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
    parser.add_argument(
        "--policy",
        choices=["default-no-split-k", "default", "stable", "model-spill", "joint"],
        action="append",
    )
    args = parser.parse_args()
    if args.policy:
        schedule = json.loads((args.root / "schedule.json").read_text())
        if args.policy not in schedule["pairs"]:
            parser.error("Measured arms must match an approved pair")
    policies = args.policy or ("default-no-split-k", "default", "stable", "model-spill")
    root = args.root
    os.environ["TINKER_API_KEY"] = Path("/tmp/spindle-rl-three-arm-api-key").read_text()
    config = json.loads((root / "initial-states/measured-config.json").read_text())
    config["checkpoint_each_update"] = True
    stage = "measured"
    (root / (stage + "-config.json")).write_text(json.dumps(config, indent=2))
    children = []
    try:
        for policy in policies:
            directory = root / policy
            endpoint = json.loads((directory / "endpoint.json").read_text())["base_url"]
            held = directory / "initialization-model.json"
            if not held.exists():
                service = tinker.ServiceClient(base_url=endpoint)
                trainer = service.create_lora_training_client(
                    config["model"],
                    rank=config["rank"],
                    train_attn=True,
                    train_mlp=True,
                    train_unembed=False,
                )
                held.write_text(json.dumps(trainer.model_id))
            command = [
                sys.executable,
                "-u",
                "scripts/benchmarks/lora_routing/supervise.py",
                "--root",
                str(directory / stage),
                "--base-url",
                endpoint,
                "--config",
                str(root / (stage + "-config.json")),
                "--clients",
                "8",
                "--stagger-seconds",
                str(config["client_stagger_seconds"]),
                "--hours",
                "8",
                "--initialization-timeout",
                "1800",
                "--environment",
                "lilo-deploy",
                "--keep-apps",
                "--recover-clients",
                "--resource-since",
                str(time.time()),
            ]
            command += ["--initialization-model", json.loads(held.read_text())]
            if (directory / stage).exists():
                command.append("--resume")
            command += [
                "--owned-apps",
                *json.loads((directory / "owned-apps.json").read_text()),
            ]
            with (directory / (stage + ".log")).open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            children.append((policy, child))
        while any(child.poll() is None for _, child in children):
            if (root / "STOP").exists():
                raise RuntimeError("Experiment stopped")
            # An exhausted recovery budget in one arm must not cancel its peer.
            (root / "pair-process-status.json").write_text(
                json.dumps({policy: child.poll() for policy, child in children})
            )
            time.sleep(5)
        for policy, child in children:
            if child.returncode:
                raise RuntimeError(f"{policy} {stage} failed: {child.returncode}")
        suffix = "-" + "-".join(policies) if args.policy else ""
        (root / (stage + "-complete" + suffix + ".json")).write_text(
            json.dumps(dict(time=time.time()))
        )
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            child.wait(timeout=180)


if __name__ == "__main__":
    main()
