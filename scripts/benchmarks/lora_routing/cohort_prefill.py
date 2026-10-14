"""Experimental group admission control; all prefill work stays in the timer.

This benchmark helper tests whether preparing a reusable hybrid-cache boundary
before releasing siblings is worthwhile. It is not enabled in production.
"""

import asyncio
import time
import uuid

from stitch.types import VersionRef

from spindle.inference.sampling import _cache_affinity_id
from spindle.inference.lora_routing import prefix_key
from spindle.inference.routing_work import token_prefixes


async def prepare_cohort(task, gateway, client, router, records, *, native=False):
    prompt = task["payload"]["prompt"]["chunks"][0]["tokens"]
    count = task["payload"]["num_samples"]
    boundary = (len(prompt) - 1) // 64 * 64
    if count < 2 or boundary == 0:
        return
    model, version = task["model_id"], task["publish_version"]
    adapter = VersionRef(model, version).identity
    affinity = _cache_affinity_id(task)
    reservations = []
    destinations = {}
    try:
        if native:
            await router.refresh()
            await router.prepare_prefix(adapter, affinity, prompt=prompt)
            home = router.prefix_homes.get(prefix_key(adapter, affinity))
            replica = router.replicas.get(home)
            digest = token_prefixes(adapter, prompt)[-1][0]
            if (
                replica is None
                or replica.work is None
                or digest not in replica.work["prefixes"]
            ):
                destinations[None] = {"Modal-Session-ID": affinity}
        else:
            await router.refresh()
            prefixes = await router.prepare_prefix(adapter, affinity, prompt=prompt)
            # Reserve every sibling while planning, so simultaneously preparing
            # groups do not all see an empty replica. These are local estimates.
            for index in range(count):
                rid = "prepare-" + uuid.uuid4().hex
                detail = {}
                host = router.select_destination(
                    model,
                    adapter,
                    affinity,
                    prompt=prompt,
                    output_tokens=task["payload"]["sampling_params"]["max_tokens"],
                    reservation_id=rid,
                    details=detail,
                    prefixes=prefixes,
                    group_id=task["request_id"],
                    group_size=count,
                    sample_index=index,
                )
                if host is not None:
                    reservations.append((rid, host))
                    if detail["cached_tokens"] < boundary:
                        destinations[host] = {
                            "Modal-Flash-Upstream": router.replicas[host].upstream
                        }

        async def prefill(host, headers):
            began = time.time()
            response = await client.post(
                gateway + "/generate",
                headers=headers,
                timeout=240,
                json=dict(
                    rid="cohort-" + uuid.uuid4().hex,
                    input_ids=prompt[:boundary],
                    weight_run_id=model,
                    weight_version=dict(exact_version=version, min_version=None),
                    sampling_params=dict(
                        max_new_tokens=1, ignore_eos=True, temperature=0
                    ),
                    return_logprob=False,
                ),
            )
            response.raise_for_status()
            data = response.json()
            meta = data["meta_info"]
            assert meta["weight_version_start"] == meta["weight_version_end"] == version
            actual = response.headers.get("x-spindle-replica")
            assert host is None or actual == host
            records.append(
                dict(
                    group=task["request_id"],
                    adapter=adapter,
                    replica=actual,
                    start=began,
                    end=time.time(),
                    prompt_tokens=boundary,
                    output_tokens=len(data["output_ids"]),
                    meta=meta,
                )
            )

        pending = [
            asyncio.create_task(prefill(h, headers))
            for h, headers in destinations.items()
        ]
        try:
            await asyncio.gather(*pending)
        except BaseException:
            # gather propagates an individual failure without cancelling its
            # siblings. Drain their cancellation before releasing reservations.
            for request in pending:
                request.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
    finally:
        for rid, host in reservations:
            router.release(host, adapter, reservation_id=rid)
