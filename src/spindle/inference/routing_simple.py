"""Simple routing controls sharing the same observations and reservations.

These deliberately omit the measured-work score. They test whether request
counts and a small amount of locality are sufficient, before attributing gains
to the more complex work/domain policy.
"""

from .routing_domains import DomainRouter


SIMPLE_POLICIES = (
    "least-load",
    "adapter-home",
    "adapter-two",
    "adapter-spill",
    "balanced-home",
    "balanced-spill",
    "model-home",
    "model-spill",
    "model-wave",
    "model-idle",
    "model-locality",
    "bounded-kv",
    "guarded-locality",
    "guarded-least-load",
    "guarded-home",
)


class SimpleRouter(DomainRouter):
    def __init__(self, *args, policy, **kwargs):
        if policy not in SIMPLE_POLICIES:
            raise ValueError(f"Unknown simple routing policy: {policy}")
        super().__init__(*args, balanced=True, stable=True, **kwargs)
        self.guarded = policy.startswith("guarded-")
        self.policy = {
            "guarded-locality": "model-locality",
            "guarded-least-load": "least-load",
            "guarded-home": "model-home",
        }.get(policy, policy)
        if self.policy.startswith(("balanced-", "model-")):
            # Round-robin adapter counts with rendezvous tie breaking. Unlike
            # the original slot-budget map this also balances a small catalog.
            self.even_primaries = True

    def select_destination(self, *args, **kwargs):
        if self.guarded and any(
            r.work is not None and not r.work.get("admission_supported", False)
            for r in self.replicas.values()
        ):
            return None
        return super().select_destination(*args, **kwargs)

    def placement_identity(self, adapter):
        # Placement may survive an optimizer publication. Cache matches, weight
        # selection, residency, and reservations still use the exact version.
        return (
            adapter.rsplit("/", 1)[0] if self.policy.startswith("model-") else adapter
        )

    def request_count(self, host):
        replica = self.replicas[host]
        observed = replica.work["request_ids"]
        return replica.active + sum(
            rid not in observed for rid in self.host_reservations.get(host, {})
        )

    def reuse_group_route(self, host):
        return (
            not self.guarded or self.request_count(host) < self.rates.decode_concurrency
        )

    def placement_candidates(self, model, adapter, candidates, count):
        loads = {}
        for host, candidate in candidates.items():
            loads[host] = self.request_count(host)
            candidate["observed_and_reserved_requests"] = loads[host]
            candidate["score"] = float(loads[host])
        if self.guarded:
            capacity = self.rates.decode_concurrency
            spare = {h: c for h, c in candidates.items() if loads[h] < capacity}
            # Once the observed pool is full, keep ordinary placement and queue
            # at the engine. Admission probes are only useful while another
            # destination appears to have room; they are not a global semaphore.
            if spare:
                candidates = spare
                for candidate in candidates.values():
                    candidate["admission_limit"] = capacity
        if len(candidates) == 1 or self.policy == "least-load":
            return candidates
        if self.policy == "model-locality":
            # Separate load feasibility from state-reconstruction cost. This
            # avoids adding a noisy drain-time estimate to cache/setup cost.
            # One engine batch bounds the observed request-count preference;
            # it is not a distributed admission or completion-time guarantee.
            limit = min(loads.values()) + self.rates.decode_concurrency
            ranking = self.primary_placement(adapter, self.placement_topology(adapter))
            eligible = {}
            for host in ranking:
                if host not in candidates or loads[host] > limit:
                    continue
                candidate = candidates[host]
                candidate["score"] = (
                    candidate["registration_seconds"]
                    + candidate["transfer_seconds"]
                    + candidate["group_prefill_seconds"]
                )
                candidate["request_count_limit"] = limit
                eligible[host] = candidate
            return eligible
        limit = 1.25 * sum(loads.values()) / len(loads) + count
        if self.policy in {
            "adapter-home",
            "adapter-two",
            "adapter-spill",
            "balanced-home",
            "balanced-spill",
            "model-home",
            "model-spill",
            "model-wave",
            "model-idle",
        }:
            hosts = self.placement_topology(adapter)
            ranking = self.primary_placement(adapter, hosts)
            available = [h for h in ranking if h in candidates]
            home = available[0]
            capacity = self.rates.decode_concurrency
            if self.policy == "model-wave":
                # Keep locality while its queue is within one engine batch of
                # the least-loaded destination. This is a request-count budget,
                # not a claim about equal completion times for unequal lengths.
                limit = min(loads.values()) + capacity
                return {home: candidates[home]} if loads[home] <= limit else candidates
            if self.policy == "model-idle":
                # Require observed pressure at home before breaking placement;
                # local reservations alone cannot establish remote congestion.
                spare = {h: c for h, c in candidates.items() if loads[h] < capacity}
                if self.replicas[home].active < capacity or not spare:
                    return {home: candidates[home]}
                return spare
            if self.policy.endswith("-spill"):
                return {home: candidates[home]} if loads[home] <= limit else candidates
            width = 2 if self.policy == "adapter-two" else 1
            return {h: candidates[h] for h in available[:width]}
        # A classic bounded-load locality control: retain the best actual KV
        # match only among replicas within 25% of mean load plus one chunk.
        # The chunk allowance is explicit, rather than hidden in a score term.
        eligible = {h: c for h, c in candidates.items() if loads[h] <= limit}
        longest = max(c["cached_tokens"] for c in eligible.values())
        return {h: c for h, c in eligible.items() if c["cached_tokens"] == longest}
