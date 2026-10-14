"""CPU cost of routing against synthetic snapshots; no GPU scaling claim."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_domains import DomainRouter
from spindle.inference.routing_work import WorkRates, WorkRouter, token_prefixes


async def empty():
    return []


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    prompt = list(range(8192))
    adapters = [f"model-{i}/weight_v000004" for i in range(64)]
    prefixes = {a: token_prefixes(a, prompt) for a in adapters}
    for replica_count in (8, 32, 128):
        for policy in ("work", "stable"):
            for group in (1, 8):
                router = (WorkRouter if policy == "work" else DomainRouter)(
                    empty,
                    rates=WorkRates(8706, 1065, 0.067, 1.87, 36),
                    **({} if policy == "work" else dict(balanced=True, stable=True)),
                )
                router.replicas = {
                    f"host-{i}": Replica(
                        f"host-{i}",
                        active=36,
                        adapters={adapters[(i * 8 + j) % 64]: 4 for j in range(8)},
                        work=dict(
                            age_seconds=0,
                            prefixes=frozenset(
                                h for a in adapters[i % 8 :: 8] for h, n in prefixes[a]
                            ),
                            prefill_tokens=36 * 1024,
                            decode_tokens=36 * 256,
                            pending_adapters=adapters[i % 8 :: 8],
                            request_ids=frozenset(f"observed-{j}" for j in range(36)),
                            cpu_adapters=adapters,
                            gpu_adapters=adapters[i % 8 :: 8],
                        ),
                    )
                    for i in range(replica_count)
                }
                elapsed = []
                reservations = []
                for index in range(1200):
                    adapter = adapters[(index // group) % 64]
                    router.updated = time.monotonic()
                    began = time.perf_counter()
                    host = router.select_destination(
                        adapter.split("/")[0],
                        adapter,
                        str(index // group),
                        prompt=prompt,
                        prefixes=prefixes[adapter],
                        output_tokens=512,
                        reservation_id=str(index),
                        group_id=str(index // group),
                        group_size=group,
                        sample_index=index % group,
                    )
                    elapsed.append((time.perf_counter() - began) * 1e6)
                    reservations.append((host, adapter, str(index)))
                    if len(reservations) > 128:
                        h, a, rid = reservations.pop(0)
                        router.release(h, a, reservation_id=rid)
                values = elapsed[200:]
                results.append(
                    dict(
                        policy=policy,
                        replicas=replica_count,
                        group=group,
                        requests=len(values),
                        mean_us=statistics.mean(values),
                        p50_us=float(np.percentile(values, 50)),
                        p95_us=float(np.percentile(values, 95)),
                        p99_us=float(np.percentile(values, 99)),
                    )
                )
    hashing = []
    for _ in range(1000):
        began = time.perf_counter()
        token_prefixes(adapters[0], prompt)
        hashing.append((time.perf_counter() - began) * 1e6)
    args.output.write_text(
        json.dumps(
            dict(
                rows=results,
                prefix_hash_p50_us=float(np.percentile(hashing, 50)),
                prefix_hash_p95_us=float(np.percentile(hashing, 95)),
            ),
            indent=2,
        )
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
