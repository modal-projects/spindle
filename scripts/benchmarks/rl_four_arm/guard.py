"""Notify the arm supervisor when its serving trainer exits; preserve other arms."""

import json
from pathlib import Path
import sys
import time

from scripts.benchmarks.lora_routing.recovery import atomic_json


def inspect(root, trainers):
    resources = {
        r["task_id"]: r for r in json.loads((root / "gpu-lifetimes.json").read_text())
    }
    counts = {}
    for policy, trainer in trainers.items():
        directory = root / policy / "measured"
        updates = 0
        for client in directory.glob("client-*"):
            if client.is_dir():
                updates += len(list(client.glob("commit-*.json")))
        counts[policy] = updates
        if resources.get(trainer, {}).get("finished_at") and updates < 80:
            if not directory.exists():
                continue
            notice = directory / "RECOVER_TRAINER.json"
            if not notice.exists():
                atomic_json(notice, dict(time=time.time(), trainer=trainer))
            # Subsequent trainer incarnations are recovered by client RPC failure
            # and SDK deadlines. Never infer failure from an unrelated spare task.
    return counts


def main():
    root = Path(sys.argv[1])
    trainers = json.loads((root / "trainer-incarnations.json").read_text())
    while not (root / "STOP").exists():
        counts = inspect(root, trainers)
        atomic_json(root / "guard-status.json", dict(time=time.time(), updates=counts))
        if all(n == 80 for n in counts.values()):
            atomic_json(
                root / "guard-complete.json", dict(time=time.time(), updates=counts)
            )
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
