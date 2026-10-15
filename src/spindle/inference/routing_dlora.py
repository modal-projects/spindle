"""Decentralized demand-driven replication with measured warm-wait comparison.

Workers share only read-only replica observations. Existing prefix and pending
work estimates are expressed in seconds, not weighted cache/load bonuses.
Replica sidecars alone own merge mode and enforce admission capacity.
"""

from .routing_work import WorkRouter


class DloraRouter(WorkRouter):
    def reuse_group_route(self, host):
        state = self.replicas[host]
        observed = state.work["request_ids"]
        outstanding = state.active + sum(
            rid not in observed for rid in self.host_reservations.get(host, {})
        )
        return outstanding < self.rates.decode_concurrency

    def placement_candidates(self, model, adapter, candidates, count):
        eligible = {}
        for host, candidate in candidates.items():
            mode = self.replicas[host].work.get("mode") or {}
            if mode.get("failed") or mode.get("transitioning"):
                continue
            merged = mode.get("merged_adapter")
            # Avoid forcing a busy specialized engine through a drain and two
            # cache flushes when another engine can serve the adapter directly.
            if merged is not None and merged != adapter and mode.get("running", 0):
                continue
            profile = mode.get("profile")
            if merged is not None and merged != adapter and profile:
                candidate["unmerge_seconds"] = profile["unmerge_seconds"]
                candidate["score"] += profile["unmerge_seconds"]
            if merged == adapter and profile:
                ratio = (
                    profile["merged_token_seconds"] / profile["unmerged_token_seconds"]
                )
                queued_decode = self.replicas[host].work["decode_tokens"]
                observed = self.replicas[host].work["request_ids"]
                queued_decode += sum(
                    r["decode_tokens"]
                    for rid, r in self.host_reservations.get(host, {}).items()
                    if rid not in observed
                )
                own_decode = (
                    candidate["placement_group_size"]
                    * candidate["requested_decode_tokens"]
                )
                saving = (
                    (queued_decode + own_decode)
                    / self.rates.decode_tokens_per_second
                    * (1 - ratio)
                )
                candidate["score"] -= saving
                candidate["merged_decode_saving_seconds"] = saving
            eligible[host] = candidate
        # All-specialized pools must still serve new versions/cold adapters.
        # The owning sidecar will drain and unmerge before admitting that work.
        return eligible or candidates
