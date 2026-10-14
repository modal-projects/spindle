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
    "bounded-kv",
)


class SimpleRouter(DomainRouter):
    def __init__(self, *args, policy, **kwargs):
        if policy not in SIMPLE_POLICIES:
            raise ValueError(f"Unknown simple routing policy: {policy}")
        super().__init__(*args, balanced=True, stable=True, **kwargs)
        self.policy = policy
        if policy.startswith(("balanced-", "model-")):
            # Round-robin adapter counts with rendezvous tie breaking. Unlike
            # the original slot-budget map this also balances a small catalog.
            self.even_primaries = True

    def placement_identity(self, adapter):
        # Placement may survive an optimizer publication. Cache matches, weight
        # selection, residency, and reservations still use the exact version.
        return (
            adapter.rsplit("/", 1)[0] if self.policy.startswith("model-") else adapter
        )

    def placement_candidates(self, model, adapter, candidates, count):
        loads = {}
        for host, candidate in candidates.items():
            replica = self.replicas[host]
            observed = replica.work["request_ids"]
            local = sum(
                rid not in observed for rid in self.host_reservations.get(host, {})
            )
            loads[host] = replica.active + local
            candidate["observed_and_reserved_requests"] = loads[host]
            candidate["score"] = float(loads[host])
        if len(candidates) == 1 or self.policy == "least-load":
            return candidates
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
