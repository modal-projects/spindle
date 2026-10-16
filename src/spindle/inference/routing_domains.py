"""Experimental demand-sized adapter domains with bounded work imbalance.

The preferred replica set grows by one engine-sized batch of outstanding demand.
Exact-version cache matches remain eligible outside that set. A one-wave work
budget bounds the locality preference when other adapters congest a domain.
All decisions use local snapshots; this is not a distributed admission limit.
"""

import math
import time

from .lora_routing import _ranking
from .routing_work import WorkRouter


class DomainRouter(WorkRouter):
    def __init__(self, *args, balanced=False, stable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.balanced = balanced
        self.stable = stable
        self.even_primaries = False
        self.catalog_key = None
        self.primary_hosts = {}
        self.host_seen = {}
        self.adapter_seen = {}

    def placement_topology(self, adapter):
        now = time.monotonic()
        for host, replica in self.replicas.items():
            self.host_seen[host] = self.updated
            if replica.work:
                for name in (
                    replica.work["cpu_adapters"] + replica.work["pending_adapters"]
                ):
                    self.adapter_seen[name] = self.updated
        self.adapter_seen[adapter] = now
        self.host_seen = {
            h: t for h, t in self.host_seen.items() if now - t < self.stale_after
        }
        self.adapter_seen = {
            a: t for a, t in self.adapter_seen.items() if now - t < self.stale_after
        }
        self.adapter_seen.pop("base", None)
        return tuple(sorted(self.host_seen))

    def primary_placement(self, adapter, hosts):
        catalog = {adapter}
        for replica in self.replicas.values():
            if replica.work:
                catalog.update(replica.work["cpu_adapters"])
                catalog.update(replica.work["pending_adapters"])
        catalog.discard("base")
        if self.stable:
            catalog.update(self.adapter_seen)
        catalog = {self.placement_identity(a) for a in catalog}
        key = (hosts, frozenset(catalog))
        if key != self.catalog_key:
            assigned = dict.fromkeys(hosts, 0)
            self.primary_hosts = {}
            for name in sorted(catalog):
                rank = _ranking(name, hosts)
                # Fill each replica's adapter-slot budget before placing an
                # additional budget's worth anywhere. With <= N*S catalog
                # entries this yields at most S preferred primaries per replica.
                host = min(
                    rank,
                    key=lambda h: assigned[h]
                    if self.even_primaries
                    else assigned[h] // self.slots,
                )
                assigned[host] += 1
                self.primary_hosts[name] = host
            self.catalog_key = key
        primary = self.primary_hosts[self.placement_identity(adapter)]
        return (primary,) + tuple(h for h in _ranking(adapter, hosts) if h != primary)

    def placement_identity(self, adapter):
        return adapter

    def placement_candidates(self, model, adapter, candidates, count):
        if len(candidates) <= 1:
            return candidates
        demand = count
        for host, replica in self.replicas.items():
            demand += replica.adapters.get(adapter, 0)
            observed = replica.work["request_ids"] if replica.work else ()
            demand += sum(
                r["adapter"] == adapter and rid not in observed
                for rid, r in self.host_reservations.get(host, {}).items()
            )
        capacity = self.rates.decode_concurrency
        width = min(len(candidates), max(1, math.ceil(demand / capacity)))
        # The balanced map places exact versions independently. The basic
        # policy ranks by model identity to preserve placement across updates.
        # All cache and residency credit requires the exact adapter version.
        # Eligibility can change because of one slow observation or one failed
        # request. Do not rebuild every adapter's placement on that transient
        # subset. Expire topology/catalog entries after the normal stale horizon;
        # an ineligible primary spills only its own traffic in the meantime.
        hosts = (
            self.placement_topology(adapter)
            if self.stable
            else tuple(sorted(candidates))
        )
        ranking = (
            self.primary_placement(adapter, hosts)
            if self.balanced
            else _ranking(model, hosts)
        )
        domain = set(ranking[:width])
        best = min(c["score"] for c in candidates.values())
        # One full engine batch of this request's decode plus cold prefill is a
        # dimensioned, workload-dependent escape budget, not a tuned cache bonus.
        wave = capacity * max(c["work"] for c in candidates.values())
        eligible = {
            h: c
            for h, c in candidates.items()
            if (h in domain or c["cached_tokens"] > 0) and c["score"] <= best + wave
        }
        for c in candidates.values():
            c["domain_width"] = width
            c["adapter_demand"] = demand
            c["escape_budget_seconds"] = wave
        return eligible or candidates
