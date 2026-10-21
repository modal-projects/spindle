# Inference scheduling

Set `inference_lora_routing = "model-spill"` in a deployment recipe to enable
model placement with overflow. The default remains `"affinity"`, using Flash's
existing routing. The sampling API and exact-version checks are unchanged.

## Model placement with overflow

Each sampling worker keeps one router per inference pool, shared across models.
The policy builds a deterministic, balanced preferred-replica map over observed
model identities. Adapter versions share a placement identity; affinity and
sample-group history include the exact adapter version.

A generation group is divided into chunks of at most `max_running_requests`
samples (or the inference target concurrency when that setting is absent).
Siblings in a chunk share a destination. For a new chunk, the preferred replica
is eligible for the locality preference when its load is at most
`1.25 * mean_pool_load + chunk_size`. Otherwise the chunk goes to the least-loaded
eligible replica. Prior exact-version affinity breaks load ties. This retains
the evaluated policy's heuristic; it is not a distributed admission bound.

Loads count observed active requests plus this worker's reservations that have
not appeared in the snapshot. Completed, failed, and cancelled attempts release
their reservations. A rejection or transport failure temporarily excludes that
replica, and the retry selects independently of the original sibling chunk.
Native SGLang admission still decides whether a replica accepts a request.

The policy preserves affinity as a locality hint. It does not inspect KV trees,
measure maximal prefix reuse, or prove that an adapter remains GPU-resident.
Catalog changes can alter placement; a ten-second topology/catalog history
prevents a transient missed observation from immediately remapping other models.

## Extension interface

`SchedulingPolicy.select_engine(request, pool)` is synchronous and returns an
engine ID from `pool.loads`, or `None` to delegate to native Flash routing.
`RoutingRequest` carries exact adapter identity, affinity, and sibling-group
coordinates. `PoolSnapshot.engines` contains observations, while `pool.loads`
contains only fresh, non-failed candidates with local reservations accounted for.
Policies must not perform I/O or mutate the snapshot.

`InferenceRouter` owns observation refresh, eligibility, reservations, and
`finish_request(request_id, failed=...)`. Selection and reservation execute
without an `await` on the sampling worker's event loop. There is no central
controller or cross-worker lock. A new policy implements `select_engine` and is
passed to the Modal provider's `create_router`; transport and retry handling in
`sample_task` stay shared. Wire its recipe choice at the deployment boundary.

Discovery uses the authenticated Flash gateway with `Modal-Flash-Upstream` to
address replicas. Membership is cached for five seconds. Active routers poll
replicas concurrently once per second with a three-second refresh deadline.
Fresh requests use the local snapshot; cold/stale callers share one refresh.
Unusable observations fall back to the existing Flash affinity path, which also
allows new replicas to receive traffic before discovery catches up. Explicit
routes omit conflicting Flash affinity headers.

The SGLang hook publishes only its CPU adapter catalog, at most once a second,
through a bounded asynchronous file writer. The sidecar adds its active request
IDs and pending adapters. No KV-tree export or benchmark endpoint is required.
The hook supports TP with DP=1 and PP=1; other layouts use native routing.

## Validation boundary

This PR preserves the model-spill algorithm from the serving experiments and
separates its lifecycle from policy selection. It does not establish an
end-to-end RL improvement over current Spindle. Matched evaluation results will
be added to the PR separately.
