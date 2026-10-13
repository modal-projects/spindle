"""Compare projected drain work on the inference scheduler.

Rates are measured for the deployed model/hardware, not dimensionless weights.
This is an admission heuristic, not an exact continuous-batching simulator.
"""

from array import array
import asyncio
from collections import Counter, OrderedDict
from dataclasses import dataclass
import math
import time

import xxhash

from .lora_routing import LoraRouter, _ranking, prefix_key


def token_prefixes(adapter: str, tokens: list[int]) -> tuple[tuple[str, int], ...]:
    """Version-scoped prefix hashes at conservative 64-token boundaries.

    The last input token must be evaluated to produce next-token logits. Only
    boundaries also advertised by the engine receive cache credit.
    """
    digest = xxhash.xxh3_128((adapter + "\0").encode())
    result = []
    values = array("q", tokens)
    for end in range(64, len(tokens), 64):
        digest.update(values[end - 64 : end].tobytes())
        result.append((digest.hexdigest(), end))
    return tuple(result)


@dataclass(frozen=True)
class WorkRates:
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    gpu_lora_seconds: float
    registration_seconds: float
    decode_concurrency: int = 1

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("routing service rates and delays must be positive")
        if int(self.decode_concurrency) != self.decode_concurrency:
            raise ValueError("decode concurrency must be an integer")


class WorkRouter(LoraRouter):
    def __init__(self, *args, rates: WorkRates, adapter_costs: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rates = rates
        self.adapter_costs = adapter_costs
        self.reservations = {}
        self.host_reservations = {}
        self.group_routes = OrderedDict()

    async def refresh(self):
        await super().refresh()
        now = time.monotonic()
        if (
            self.pending is not None
            and not self.pending.done()
            and not any(
                r.work is not None and r.work["age_seconds"] + now - self.updated < 3
                for r in self.replicas.values()
            )
        ):
            # Coalesce the first burst after engine observations expire. Normal
            # selections still use a fresh local snapshot without waiting.
            await asyncio.shield(self.pending)

    async def prepare_prefix(self, adapter, affinity, *, prompt=()):
        # Hash once per sample group, never once per replica or generation.
        return token_prefixes(adapter, prompt)

    def select_destination(
        self,
        model,
        adapter,
        affinity,
        *,
        prompt=(),
        output_tokens=1,
        reservation_id=None,
        details=None,
        prefixes=None,
        group_id=None,
        group_size=1,
        sample_index=0,
    ):
        now = time.monotonic()
        if now - self.updated > self.stale_after:
            return None
        hosts = tuple(
            sorted(
                h
                for h, r in self.replicas.items()
                if self.failed_until.get(h, 0) <= now
                and r.work is not None
                and r.work.get("age_seconds", 99) + now - self.updated < 3
            )
        )
        if not hosts:
            return None
        if reservation_id is None:
            raise ValueError("work routing requires an individual reservation ID")
        key = prefix_key(adapter, affinity)
        if prefixes is None:
            prefixes = token_prefixes(adapter, prompt)
        rates = self.rates
        capacity = rates.decode_concurrency
        chunk = sample_index // capacity
        count = min(capacity, group_size - chunk * capacity)
        group_key = (adapter, group_id, chunk) if group_id is not None else None
        planned = self.group_routes.get(group_key)
        if planned in hosts:
            hosts = (planned,)
        candidates = {}
        for host in _ranking(key, hosts):
            state = self.replicas[host].work
            cached = state["prefixes"]
            hit = next((n for digest, n in reversed(prefixes) if digest in cached), 0)
            work = state["prefill_tokens"] / rates.prefill_tokens_per_second
            work += state["decode_tokens"] / rates.decode_tokens_per_second
            pending_adapters = set(state["pending_adapters"])
            observed = state["request_ids"]
            for rid, reservation in self.host_reservations.get(host, {}).items():
                if rid not in observed:
                    work += reservation["work"]
                    pending_adapters.add(reservation["adapter"])
            gpu = set(state["gpu_adapters"])
            cpu = set(state["cpu_adapters"])
            # Each already queued cold adapter needs a transfer, not every
            # sample using it. Reloads caused by interleaving are model error;
            # record them in validation rather than inventing an eviction term.
            if self.adapter_costs:
                work += len(pending_adapters - gpu) * rates.gpu_lora_seconds
                work += len(pending_adapters - cpu) * rates.registration_seconds
            registration = 0.0
            transfer = 0.0
            if self.adapter_costs:
                if adapter not in cpu and adapter not in pending_adapters:
                    # The pinned engine registers adapters synchronously on its
                    # scheduler thread. Treat setup as service work rather than
                    # pretending it is hidden by queued inference. Volume I/O
                    # can overlap, so full exposed registration is conservative.
                    registration = rates.registration_seconds
                if adapter not in gpu and adapter not in pending_adapters:
                    transfer = rates.gpu_lora_seconds
            prefill = (len(prompt) - hit) / rates.prefill_tokens_per_second
            decode = output_tokens / rates.decode_tokens_per_second
            # Optimize a known generation group together. Greedily optimizing
            # each sibling scatters one prompt over every replica and can evict
            # the very shared prefixes we want to preserve. Large groups split
            # at the measured engine running capacity, not Flash's scale target.
            score = work + registration + transfer + count * (prefill + decode)
            candidates[host] = dict(
                score=score,
                backlog_seconds=work,
                registration_seconds=registration,
                transfer_seconds=transfer,
                prefill_seconds=prefill,
                cached_tokens=hit,
                work=prefill + decode,
                placement_group_size=count,
            )
        chosen = min(candidates, key=lambda h: candidates[h]["score"])
        # Retain locality only inside a narrow numerical tie. No arbitrary
        # target-concurrency allowance or multiplicative affinity bonus.
        home = self.prefix_homes.get(key)
        if (
            home in candidates
            and candidates[home]["score"] <= candidates[chosen]["score"] + 1e-6
        ):
            chosen = home
        self._remember_prefix(key, chosen)
        if group_key is not None:
            self.group_routes[group_key] = chosen
            self.group_routes.move_to_end(group_key)
            if len(self.group_routes) > 8192:
                self.group_routes.popitem(last=False)
        self.reservations[reservation_id] = dict(
            host=chosen, adapter=adapter, work=candidates[chosen]["work"]
        )
        self.host_reservations.setdefault(chosen, {})[reservation_id] = (
            self.reservations[reservation_id]
        )
        self.local.setdefault(chosen, Counter())[adapter] += 1
        if details is not None:
            details.update(candidates[chosen])
            details["selection_seconds"] = time.monotonic() - now
        return chosen

    def release(self, upstream, adapter, *, failed=False, reservation_id=None):
        reservation = self.reservations.pop(reservation_id, None)
        if reservation is not None:
            host = reservation["host"]
            pending = self.host_reservations[host]
            pending.pop(reservation_id)
            if not pending:
                del self.host_reservations[host]
        super().release(upstream, adapter, failed=failed)
