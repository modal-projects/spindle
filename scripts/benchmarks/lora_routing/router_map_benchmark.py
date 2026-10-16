"""Measure primary-map maintenance on CPU; this does not allocate inference GPUs."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

from spindle.inference.lora_routing import Replica, _ranking
from spindle.inference.routing_domains import DomainRouter
from spindle.inference.routing_work import WorkRates


async def discover():
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for n in (8, 32, 128):
        router = DomainRouter(
            discover,
            rates=WorkRates(8706, 1065, 0.067, 1.87, 36),
            balanced=True,
            stable=True,
        )
        catalog = [f"model-{i}/v4" for i in range(64)]
        router.replicas = {
            f"host-{i}": Replica(
                f"host-{i}", work={"cpu_adapters": catalog, "pending_adapters": []}
            )
            for i in range(n)
        }
        hosts = tuple(sorted(router.replicas))
        for mode in ("retained-map", "catalog-rebuild", "topology-rebuild"):
            elapsed = []
            for _ in range(120):
                if mode != "retained-map":
                    router.catalog_key = None
                if mode == "topology-rebuild":
                    _ranking.cache_clear()
                began = time.perf_counter()
                router.primary_placement(catalog[0], hosts)
                elapsed.append((time.perf_counter() - began) * 1e6)
            values = sorted(elapsed[20:])
            rows.append(
                dict(
                    replicas=n,
                    mode=mode,
                    p50_us=statistics.median(values),
                    p95_us=values[94],
                    max_us=max(values),
                )
            )
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                python=platform.python_version(),
                architecture=platform.machine(),
                cpu=cpu,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
