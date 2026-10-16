"""Save common initial adapter states before any measured RL comparison."""

import argparse
import json
import os
from pathlib import Path

import httpx

from client import Client, read_config
from supervise import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--unload-stale-model", action="append", default=[])
    args = parser.parse_args()
    args.root.mkdir(exist_ok=False)
    config = read_config(args.config)
    config.pop("initial_state_paths", None)
    config["save_initial_state"] = True
    paths = json.loads(args.resume_from.read_text()) if args.resume_from else []
    first = len(paths)
    if first >= args.clients:
        raise ValueError("Preparation already contains every requested client")
    for index in range(first, args.clients):
        client = Client(config, args.root / f"client-{index:02d}", args.base_url, index)
        paths.append(client.initial_state_path)
        if index == first:
            # Unloading the final model immediately scales the trainer down.
            # Keep one initialization model until a measured client attaches;
            # the supervisor then releases it before launching other clients.
            initialization_client = client
            for model_id in args.unload_stale_model:
                response = httpx.post(
                    args.base_url + "/api/v1/unload_model",
                    headers={"x-api-key": os.environ["TINKER_API_KEY"]},
                    json={"model_id": model_id},
                    timeout=120,
                )
                response.raise_for_status()
        elif not client.unload():
            raise RuntimeError(
                "Initial-state preparation could not release its training slot"
            )
        write(args.root / "initial-state-paths.json", paths)
    write(
        args.root / "initialization-model.json", initialization_client.trainer.model_id
    )
    measured = json.loads(args.config.read_text())
    measured["save_initial_state"] = False
    measured["initial_state_paths"] = paths
    write(args.root / "measured-config.json", measured)


if __name__ == "__main__":
    main()
