"""CPU cost of routing against synthetic snapshots; no GPU scaling claim."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import numpy as np

from spindle.inference.lora_routing import Replica
from spindle.inference.routing_domains import DomainRouter
from spindle.inference.routing_capacity import CapacityRouter
from spindle.inference.routing_simple import SIMPLE_POLICIES, SimpleRouter
from spindle.inference.routing_work import WorkRates, WorkRouter, token_prefixes


async def empty():
    return []


def make_router(policy, tie_seed=None):
    options = dict(rates=WorkRates(8706, 1065, 0.067, 1.87, 36))
    if policy == "stable":
        return DomainRouter(empty, balanced=True, stable=True, **options)
    if policy in SIMPLE_POLICIES:
        return SimpleRouter(empty, policy=policy, **options)
    if policy == "capacity-random":
        return CapacityRouter(empty, tie_seed=tie_seed or "cpu-benchmark", **options)
    return (CapacityRouter if policy == "capacity" else WorkRouter)(empty, **options)


def correlated_burst(policy="stable", *, warm=False):
    """Exercise correlated decisions without exchanging the local reservations."""
    loads = Counter({f"h{i}": 0 for i in range(8)})
    adapter = "hot/weight_v000004"
    prompt = list(range(8192))
    prefixes = token_prefixes(adapter, prompt)
    for frontend in range(8):
        router = make_router(policy, tie_seed=f"frontend-{frontend}")
        router.updated = time.monotonic()
        router.replicas = {
            host: Replica(
                host,
                work=dict(
                    admission_supported=True,
                    age_seconds=0,
                    prefixes=frozenset(),
                    prefill_tokens=0,
                    decode_tokens=0,
                    adapter_decode_maxima={},
                    pending_adapters=[],
                    request_ids=frozenset(),
                    cpu_adapters=[],
                    gpu_adapters=[],
                ),
            )
            for host in loads
        }
        if warm:
            router.replicas["h0"].work.update(
                prefixes=frozenset(h for h, _ in prefixes),
                cpu_adapters=[adapter],
                gpu_adapters=[adapter],
            )
        for index in range(36):
            host = router.select_destination(
                "hot",
                adapter,
                f"affinity-{frontend}",
                prompt=prompt,
                prefixes=prefixes,
                output_tokens=256,
                reservation_id=f"{frontend}-{index}",
                group_id=f"group-{frontend}",
                group_size=36,
                sample_index=index,
            )
            loads[host] += 1
    return dict(
        kind="CPU-only deterministic stale-view counterexample; no service-time or GPU measurement",
        policy=policy,
        initially_warm_replica="h0" if warm else None,
        frontends=8,
        replicas=8,
        group_size=36,
        capacity_per_replica=36,
        loads=dict(loads),
        total=sum(loads.values()),
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correlated-burst", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["work", "stable"],
        choices=[
            "work",
            "stable",
            "capacity",
            "capacity-random",
            *SIMPLE_POLICIES,
        ],
    )
    parser.add_argument(
        "--burst-policy",
        default="stable",
        choices=[
            "work",
            "stable",
            "capacity",
            "capacity-random",
            *SIMPLE_POLICIES,
        ],
    )
    args = parser.parse_args()
    if args.correlated_burst:
        result = correlated_burst(args.burst_policy, warm=args.warm)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    results = []
    prompt = list(range(8192))
    adapters = [f"model-{i}/weight_v000004" for i in range(64)]
    prefixes = {a: token_prefixes(a, prompt) for a in adapters}
    for replica_count in (8, 32, 128):
        for policy in args.policies:
            for group in (1, 8):
                router = make_router(policy)
                router.replicas = {
                    f"host-{i}": Replica(
                        f"host-{i}",
                        active=36,
                        adapters={adapters[(i * 8 + j) % 64]: 4 for j in range(8)},
                        work=dict(
                            admission_supported=True,
                            age_seconds=0,
                            prefixes=frozenset(
                                h for a in adapters[i % 8 :: 8] for h, n in prefixes[a]
                            ),
                            prefill_tokens=36 * 1024,
                            decode_tokens=36 * 256,
                            adapter_decode_maxima={
                                a: 256 for a in adapters[i % 8 :: 8]
                            },
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
