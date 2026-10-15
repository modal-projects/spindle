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
            profile = mode.get("profile")
            if profile and merged is None and profile.get("max_batch_adapters"):
                admitted = mode.get("admission_adapters", ())
                work = self.replicas[host].work
                maxima = work.get("adapter_decode_maxima", {})
                if (
                    adapter not in admitted
                    and len(admitted) >= profile["max_batch_adapters"]
                    and all(a in maxima for a in admitted)
                ):
                    # An additional adapter cannot join the native batch until
                    # one admitted cohort releases its slot. This is a service
                    # dependency, not the much smaller GPU weight-copy cost.
                    wait = (
                        min(maxima[a] for a in admitted)
                        * profile["unmerged_token_seconds"]
                    )
                    candidate["adapter_slot_wait_seconds"] = wait
                    candidate["score"] += max(0, wait - candidate["backlog_seconds"])
                    candidate["backlog_seconds"] = max(
                        wait, candidate["backlog_seconds"]
                    )
            if merged is not None and merged != adapter and profile:
                # A specialized engine is reclaimable even while busy. A hard
                # exclusion strands all other adapters on the last mixed
                # engine, regardless of its queue. Account for the drain barrier
                # in seconds and let the ordinary completion comparison decide.
                work = self.replicas[host].work
                remaining = work.get("adapter_decode_maxima", {}).get(
                    merged, work["decode_tokens"]
                )
                drain = remaining * profile["merged_token_seconds"]
                candidate["drain_seconds"] = drain
                candidate["score"] += max(0, drain - candidate["backlog_seconds"])
                candidate["backlog_seconds"] = max(drain, candidate["backlog_seconds"])
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
        # The owner drains and restores before executing incompatible work.
        return eligible
