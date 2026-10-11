"""Measure only local routing CPU time; this does not measure inference gains."""

import time
from unittest.mock import AsyncMock

from spindle.inference.lora_routing import LoraRouter, Replica


def main():
    for replicas in (8, 32):
        router = LoraRouter(AsyncMock(), target=64, slots=8)
        router.replicas = {
            str(i): Replica(
                str(i),
                active=128,
                adapters={f"model-{j}/v1": 32 for j in range(i % 4, i % 4 + 4)},
            )
            for i in range(replicas)
        }
        router.updated = time.monotonic()
        count = 100_000
        started = time.perf_counter()
        for i in range(count):
            model = f"model-{i % 16}"
            adapter = model + "/v1"
            route = router.acquire(model, adapter, f"request-{i}")
            router.release(route, adapter)
        elapsed = time.perf_counter() - started
        print(
            f"{replicas} replicas: {count} acquire/release pairs, {elapsed / count * 1e6:.2f} us/request"
        )


if __name__ == "__main__":
    main()
