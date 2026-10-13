"""Build a routing replay from completed independent DAPO training clients."""

import argparse
import gzip
import json
from pathlib import Path

from replay import load_clients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--bulletin-volume", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clients = load_clients(args.source, args.bulletin_volume)
    folders = {}
    for folder in sorted(args.source.glob("client-*")):
        if not folder.is_dir():
            continue
        events = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        model = next(e["model_id"] for e in events if e["event"] == "client_created")
        start = next(
            e["time"]
            for e in events
            if e["event"] == "train_start" and e["update"] == 2
        )
        end = next(
            e["time"] for e in events if e["event"] == "publish" and e["update"] == 2
        )
        folders[model] = (folder, end - start)
    for index, client in enumerate(clients):
        folder, duration = folders[client["model_id"]]
        client.update(
            client_index=index,
            update_seconds=duration,
            training_source=str(folder),
            prompts=[
                json.load(gzip.open(folder / f"group-{i:06d}.json.gz", "rt"))["prompt"]
                for i in range(64)
            ],
        )
    args.output.write_text(json.dumps(clients) + "\n")


if __name__ == "__main__":
    main()
