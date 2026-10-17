"""Start the other two isolated trainers while common checkpoints are prepared."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import tinker


def prepare(root, policy, config):
    directory = root / policy
    held = directory / "initialization-model.json"
    if held.exists():
        return
    endpoint = json.loads((directory / "endpoint.json").read_text())["base_url"]
    print(
        json.dumps(dict(policy=policy, event="create_start", time=time.time())),
        flush=True,
    )
    service = tinker.ServiceClient(base_url=endpoint)
    trainer = service.create_lora_training_client(
        config["model"],
        rank=config["rank"],
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    held.write_text(json.dumps(trainer.model_id))
    print(
        json.dumps(dict(policy=policy, event="create_complete", time=time.time())),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--policy", action="append")
    args = parser.parse_args()
    root = args.root
    os.environ["TINKER_API_KEY"] = Path("/tmp/spindle-rl-three-arm-api-key").read_text()
    config = json.loads((root / "config.json").read_text())
    policies = args.policy or ("default-no-split-k", "default", "stable", "model-spill")
    with ThreadPoolExecutor(max_workers=len(policies)) as executor:
        futures = [
            executor.submit(prepare, root, policy, config) for policy in policies
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
