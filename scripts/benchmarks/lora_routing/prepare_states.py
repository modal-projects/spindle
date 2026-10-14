"""Save common initial adapter states before any measured RL comparison."""

import argparse
import json
from pathlib import Path

from client import Client, read_config
from supervise import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--clients", type=int, default=16)
    args = parser.parse_args()
    args.root.mkdir(exist_ok=False)
    config = read_config(args.config)
    config.pop("initial_state_paths", None)
    config["save_initial_state"] = True
    paths = []
    for index in range(args.clients):
        client = Client(config, args.root / f"client-{index:02d}", args.base_url, index)
        paths.append(client.initial_state_path)
        # Initialization is outside every measured arm. Release the training
        # slot so all arms can create and restore through the same SDK path.
        if not client.unload():
            raise RuntimeError(
                "Initial-state preparation could not release its training slot"
            )
        write(args.root / "initial-state-paths.json", paths)
    measured = json.loads(args.config.read_text())
    measured["save_initial_state"] = False
    measured["initial_state_paths"] = paths
    write(args.root / "measured-config.json", measured)


if __name__ == "__main__":
    main()
