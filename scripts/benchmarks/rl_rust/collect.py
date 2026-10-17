"""Use the established read-only observer with this experiment's two pools."""

import argparse
import json
from pathlib import Path
import modal
from scripts.benchmarks.rl_four_arm.collect import app, collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    pools = {
        p: json.loads((args.root / p / "owned-apps.json").read_text())[-1]
        for p in json.loads((args.root / "schedule.json").read_text())["pairs"][0]
    }
    with (
        modal.enable_output(),
        app.run(),
        (args.root / "observations.jsonl").open("a") as output,
    ):
        for row in collect.remote_gen(pools, 64800):
            output.write(json.dumps(row) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
