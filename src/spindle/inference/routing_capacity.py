"""Estimate the decode iterations imposed by sequence and adapter batch limits.

For known remaining lengths, three necessary iteration counts are the longest
request, total token work divided by sequence capacity, and the sum of each
adapter's longest request divided by adapter capacity. Their maximum is a lower
bound on iterations, not a prediction guarantee for a nonpreemptive scheduler.
Natural-EOS generation substitutes the remaining token budget for unknown work.
"""

import math

from .lora_routing import _ranking
from .routing_work import WorkRouter


def decode_iterations(
    total_tokens, adapter_maxima, sequence_capacity, adapter_capacity
):
    return max(
        max(adapter_maxima.values(), default=0),
        math.ceil(total_tokens / sequence_capacity),
        math.ceil(sum(adapter_maxima.values()) / adapter_capacity),
    )


class CapacityRouter(WorkRouter):
    def __init__(self, *args, tie_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tie_seed = tie_seed

    def select_destination(self, *args, **kwargs):
        # Mixed or older observer schemas must not silently run a different
        # heuristic under this policy's name. Native Flash remains the fallback.
        if any(
            r.work is not None and "adapter_decode_maxima" not in r.work
            for r in self.replicas.values()
        ):
            return None
        return super().select_destination(*args, **kwargs)

    def placement_candidates(self, model, adapter, candidates, count):
        rates = self.rates
        step_seconds = rates.decode_concurrency / rates.decode_tokens_per_second
        for host, candidate in candidates.items():
            state = self.replicas[host].work
            total = state["decode_tokens"]
            maxima = dict(state["adapter_decode_maxima"])
            observed = state["request_ids"]
            for rid, reservation in self.host_reservations.get(host, {}).items():
                if rid in observed:
                    continue
                tokens = reservation["decode_tokens"]
                total += tokens
                name = reservation["adapter"]
                maxima[name] = max(maxima.get(name, 0), tokens)
            incoming = candidate["requested_decode_tokens"]
            total += count * incoming
            maxima[adapter] = max(maxima.get(adapter, 0), incoming)
            iterations = decode_iterations(
                total, maxima, rates.decode_concurrency, self.slots
            )
            linear_decode_seconds = total / rates.decode_tokens_per_second
            decode_seconds = iterations * step_seconds
            # Preserve the common prefill and CPU/GPU setup estimates. Replace
            # only the linear decode term that ignores batch compatibility.
            candidate["score"] += decode_seconds - linear_decode_seconds
            candidate.update(
                decode_iteration_bound=iterations,
                estimated_decode_seconds=decode_seconds,
                linear_decode_seconds=linear_decode_seconds,
                projected_adapter_count=len(maxima),
                tie_seed=self.tie_seed,
            )
        # The optional frontend seed breaks correlated cold ties. A common
        # adapter key is retained as a control. Neither choice coordinates
        # reservations across frontends or fixes a shared warm-cache hotspot.
        key = adapter if self.tie_seed is None else adapter + "\0" + self.tie_seed
        return {h: candidates[h] for h in _ranking(key, tuple(sorted(candidates)))}
