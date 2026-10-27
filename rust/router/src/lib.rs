//! One scheduler per frontend process and inference pool; native owns transport.
mod cache_wait;

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

#[derive(Deserialize)]
struct Work {
    age_seconds: f64,
    #[serde(default, deserialize_with = "deserialize_prefixes")]
    prefixes: HashSet<u128>,
    #[serde(default)]
    waiting_requests: usize,
    #[serde(default)]
    kv_used_tokens: u64,
    #[serde(default)]
    kv_capacity_tokens: u64,
    cpu_adapters: HashSet<String>,
    gpu_adapters: HashSet<String>,
    request_ids: HashSet<String>,
    pending_adapters: HashSet<String>,
    prefill_tokens: u64,
    decode_tokens: u64,
}
// Decode endpoint hashes once per observation, away from the request path.
fn deserialize_prefixes<'de, D>(deserializer: D) -> Result<HashSet<u128>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<String>::deserialize(deserializer)?
        .into_iter()
        .map(|hash| u128::from_str_radix(&hash, 16).map_err(serde::de::Error::custom))
        .collect()
}
#[derive(Deserialize)]
struct Replica {
    replica: String,
    #[serde(default)]
    upstream: String,
    active: usize,
    #[serde(default)]
    adapters: HashMap<String, usize>,
    work: Work,
}

pub struct Request {
    adapter: String,
    affinity: String,
    prompt_tokens: usize,
    output_tokens: usize,
    hashes: Vec<u128>,
    fingerprint: u128,
    preferred: Mutex<Option<String>>,
    pending_prefix: Mutex<Option<PendingPrefix>>,
}
struct PendingPrefix {
    replica: String,
    leader: String,
    started: Instant,
}

impl Request {
    fn new(
        adapter: String,
        affinity: String,
        tokens: Vec<i64>,
        output_tokens: usize,
    ) -> Result<Self, String> {
        if adapter.is_empty() || output_tokens == 0 {
            return Err(String::from("adapter and output budget are required"));
        }
        let mut hasher = Xxh3::new();
        hasher.update(adapter.as_bytes());
        hasher.update(&[0]);
        let mut hashes = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            hasher.update(&token.to_le_bytes());
            // The final prompt token must be evaluated to obtain logits.
            if (i + 1) % 64 == 0 && i + 1 < tokens.len() {
                hashes.push(hasher.digest128());
            }
        }
        Ok(Self {
            adapter,
            affinity,
            prompt_tokens: tokens.len(),
            output_tokens,
            fingerprint: hasher.digest128(),
            hashes,
            preferred: Mutex::new(None),
            pending_prefix: Mutex::new(None),
        })
    }
}

pub struct Decision {
    pub status: &'static str,
    pub replica: Option<String>,
    pub upstream: Option<String>,
    pub cached_tokens: usize,
    pub backlog_seconds: f64,
    pub prefill_seconds: f64,
    pub adapter_seconds: f64,
    pub reason: &'static str,
    pub retry_after: f64,
    pub available_replicas: usize,
    pub best_cached_tokens: usize,
}
impl Decision {
    fn wait(status: &'static str) -> Self {
        Self {
            status,
            replica: None,
            upstream: None,
            cached_tokens: 0,
            backlog_seconds: 0.0,
            prefill_seconds: 0.0,
            adapter_seconds: 0.0,
            reason: "capacity",
            retry_after: 0.5,
            available_replicas: 0,
            best_cached_tokens: 0,
        }
    }
}

#[derive(Clone)]
pub struct Candidate {
    replica_index: usize,
    adapter_tier: u8,
    preferred: bool,
    tie: u64,
    adapter_tie: u64,
    active: usize,
    adapter_active: usize,
    active_adapter_count: usize,
    queued: usize,
    kv_pinned_fraction: Option<f64>,
    cached_tokens: usize,
    backlog_seconds: f64,
    prefill_seconds: f64,
    adapter_seconds: f64,
}
struct Reservation {
    replica: String,
    adapter: String,
    work: f64,
}

/// Built away from the asyncio thread; installation only swaps immutable state.
pub struct Snapshot {
    replicas: Vec<Replica>,
    observed: Instant,
}
impl Snapshot {
    fn new(replicas: Vec<Replica>, observed: Instant) -> Result<Self, String> {
        let mut ids = HashSet::new();
        if replicas.iter().any(|r| {
            !ids.insert(&r.replica)
                || r.upstream.is_empty()
                || !r.work.age_seconds.is_finite()
                || r.work.age_seconds < 0.0
        }) {
            return Err(String::from(
                "duplicate replica, missing upstream or invalid snapshot age",
            ));
        }
        Ok(Self { replicas, observed })
    }
}
pub struct Router {
    replicas: Vec<Replica>,
    wait_policy: cache_wait::CacheWait,
    reservations: HashMap<String, Reservation>,
    groups: HashMap<String, Weak<Request>>,
    busy_until: HashMap<String, f64>,
    failed_until: HashMap<String, f64>,
    clock: Instant,
    updated: f64,
    capacity: usize,
    prefill_rate: f64,
    decode_rate: f64,
    gpu_load_seconds: f64,
    registration_seconds: f64,
}
impl Router {
    #[allow(clippy::too_many_arguments)] // Flat Python configuration boundary.
    fn new(
        policy: &str,
        capacity: usize,
        prefill_rate: f64,
        decode_rate: f64,
        gpu_load_seconds: f64,
        registration_seconds: f64,
        balance_absolute: usize,
        balance_relative: f64,
        cache_threshold: f64,
    ) -> Result<Self, String> {
        if policy != "cache_wait" {
            return Err(String::from("unknown routing policy"));
        }
        if !balance_relative.is_finite() || balance_relative < 1.0 {
            return Err(String::from("balance_relative must be finite and >= 1"));
        }
        if !cache_threshold.is_finite() || !(0.0..=1.0).contains(&cache_threshold) {
            return Err(String::from("cache_threshold must be between 0 and 1"));
        }
        let wait_policy = cache_wait::CacheWait {
            absolute: balance_absolute,
            relative: balance_relative,
            threshold: cache_threshold,
        };
        if capacity == 0
            || [prefill_rate, decode_rate]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || [gpu_load_seconds, registration_seconds]
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(String::from("invalid capacity or service rates"));
        }
        Ok(Self {
            replicas: vec![],
            wait_policy,
            reservations: HashMap::new(),
            groups: HashMap::new(),
            busy_until: HashMap::new(),
            failed_until: HashMap::new(),
            clock: Instant::now(),
            updated: 0.0,
            capacity,
            prefill_rate,
            decode_rate,
            gpu_load_seconds,
            registration_seconds,
        })
    }

    /// Siblings share pending-prefix hints only for the same exact prompt/version.
    fn prepare_group(&mut self, group: String, request: Request) -> Result<Arc<Request>, String> {
        if let Some(shared) = self.groups.get(&group).and_then(Weak::upgrade) {
            if shared.adapter != request.adapter
                || shared.affinity != request.affinity
                || shared.fingerprint != request.fingerprint
                || shared.prompt_tokens != request.prompt_tokens
                || shared.output_tokens != request.output_tokens
            {
                return Err("sampling group changed prompt, adapter or output budget".into());
            }
            return Ok(shared);
        }
        // Weak references retain no request data after completion or cancellation.
        self.groups.retain(|_, request| request.strong_count() > 0);
        let request = Arc::new(request);
        self.groups.insert(group, Arc::downgrade(&request));
        Ok(request)
    }

    /// Replace a complete observation, not an incremental request-path lookup.
    #[cfg(test)]
    fn update_snapshot(&mut self, json: &str) -> Result<(), String> {
        let observed = Instant::now();
        let rows = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
        let snapshot = Snapshot::new(rows, observed)?;
        self.install_snapshot(snapshot);
        Ok(())
    }

    fn install_snapshot(&mut self, snapshot: Snapshot) {
        self.replicas = snapshot.replicas;
        self.updated =
            self.clock.elapsed().as_secs_f64() - snapshot.observed.elapsed().as_secs_f64();
    }

    /// Select and reserve in one short call. A busy pool stays busy, not unhealthy.
    fn select_destination(
        &mut self,
        request: &Request,
        request_id: String,
        affinity_elapsed: f64,
    ) -> Result<Decision, String> {
        if !affinity_elapsed.is_finite() || affinity_elapsed < 0.0 {
            return Err(String::from("invalid affinity elapsed time"));
        }
        if self.reservations.contains_key(&request_id) {
            return Err(String::from("request already has a reservation"));
        }
        let now = self.clock.elapsed().as_secs_f64();
        let preferred = request.preferred.lock().unwrap().clone();
        let mut fresh = false;
        let mut candidates = Vec::new();
        let mut cooling_down = HashSet::new();
        for (i, replica) in self.replicas.iter().enumerate() {
            let w = &replica.work;
            if w.age_seconds + now - self.updated >= 3.0 {
                continue;
            }
            fresh = true;
            if self
                .failed_until
                .get(&replica.replica)
                .is_some_and(|t| *t > now)
            {
                continue;
            }
            if self
                .busy_until
                .get(&replica.replica)
                .is_some_and(|t| *t > now)
            {
                cooling_down.insert(i);
            }
            let unobserved: Vec<_> = self
                .reservations
                .iter()
                .filter(|(id, r)| r.replica == replica.replica && !w.request_ids.contains(*id))
                .map(|(_, r)| r)
                .collect();
            let active = replica.active + unobserved.len();
            let adapter_active = replica.adapters.get(&request.adapter).copied().unwrap_or(0)
                + unobserved
                    .iter()
                    .filter(|r| r.adapter == request.adapter)
                    .count();
            let mut active_adapters: HashSet<&str> = replica
                .adapters
                .iter()
                .filter(|(_, count)| **count > 0)
                .map(|(adapter, _)| adapter.as_str())
                .collect();
            active_adapters.extend(unobserved.iter().map(|r| r.adapter.as_str()));
            let active_adapter_count = active_adapters.len();
            // A hash covers the entire prefix and exact adapter version. The
            // longest observed endpoint proves a match without a second tree.
            let hit = request
                .hashes
                .iter()
                .enumerate()
                .rev()
                .find(|(_, hash)| w.prefixes.contains(hash))
                .map_or(0, |(index, _)| (index + 1) * 64);
            let mut pending = w.pending_adapters.clone();
            for r in &unobserved {
                pending.insert(r.adapter.clone());
            }
            let backlog = w.prefill_tokens as f64 / self.prefill_rate
                + w.decode_tokens as f64 / self.decode_rate
                + unobserved.iter().map(|r| r.work).sum::<f64>()
                + pending.difference(&w.gpu_adapters).count() as f64 * self.gpu_load_seconds
                + pending.difference(&w.cpu_adapters).count() as f64 * self.registration_seconds;
            let adapter_seconds = if pending.contains(&request.adapter) {
                0.0
            } else {
                (if w.gpu_adapters.contains(&request.adapter) {
                    0.0
                } else {
                    self.gpu_load_seconds
                }) + if w.cpu_adapters.contains(&request.adapter) {
                    0.0
                } else {
                    self.registration_seconds
                }
            };
            candidates.push(Candidate {
                replica_index: i,
                adapter_tier: if w.gpu_adapters.contains(&request.adapter) {
                    0
                } else if pending.contains(&request.adapter) {
                    1
                } else if w.cpu_adapters.contains(&request.adapter) {
                    2
                } else {
                    3
                },
                preferred: preferred.as_ref() == Some(&replica.replica),
                tie: xxh3_64(
                    format!(
                        "{}\0{}\0{}",
                        request.adapter, request.affinity, replica.replica
                    )
                    .as_bytes(),
                ),
                adapter_tie: xxh3_64(
                    format!("{}\0{}", request.adapter, replica.replica).as_bytes(),
                ),
                active,
                adapter_active,
                active_adapter_count,
                queued: w.waiting_requests,
                kv_pinned_fraction: (w.kv_capacity_tokens > 0)
                    .then(|| (w.kv_used_tokens as f64 / w.kv_capacity_tokens as f64).min(1.0)),
                cached_tokens: hit,
                backlog_seconds: backlog,
                prefill_seconds: (request.prompt_tokens - hit) as f64 / self.prefill_rate,
                adapter_seconds,
            });
        }
        let all = candidates;
        let candidates: Vec<_> = all
            .iter()
            .filter(|c| c.active < self.capacity && !cooling_down.contains(&c.replica_index))
            .cloned()
            .collect();
        let available = candidates.len();
        let best_cached = all.iter().map(|c| c.cached_tokens).max().unwrap_or(0);
        let reason;
        let selected = {
            let policy = &self.wait_policy;
            let pending = request.pending_prefix.lock().unwrap();
            let pending = pending
                .as_ref()
                .filter(|p| self.reservations.contains_key(&p.leader));
            match policy.select(
                &all,
                &candidates,
                &self.replicas,
                request,
                pending,
                self.prefill_rate,
                affinity_elapsed,
            ) {
                cache_wait::Selection::Wait(why, seconds) => {
                    let mut d = Decision::wait("wait");
                    d.reason = why;
                    d.retry_after = seconds;
                    d.available_replicas = available;
                    d.best_cached_tokens = best_cached;
                    return Ok(d);
                }
                cache_wait::Selection::Route(index, why) => {
                    reason = why;
                    index
                }
            }
        };
        let Some(selected) = selected else {
            return Ok(Decision::wait(if fresh { "wait" } else { "unavailable" }));
        };
        let c = &candidates[selected];
        let i = c.replica_index;
        let replica = &self.replicas[i];
        *request.preferred.lock().unwrap() = Some(replica.replica.clone());
        {
            let mut pending = request.pending_prefix.lock().unwrap();
            // One leader per live group; retries may replace a failed leader.
            if pending
                .as_ref()
                .is_none_or(|p| !self.reservations.contains_key(&p.leader))
            {
                *pending = Some(PendingPrefix {
                    replica: replica.replica.clone(),
                    leader: request_id.clone(),
                    started: Instant::now(),
                });
            }
        }
        self.reservations.insert(
            request_id,
            Reservation {
                replica: replica.replica.clone(),
                adapter: request.adapter.clone(),
                work: c.prefill_seconds + request.output_tokens as f64 / self.decode_rate,
            },
        );
        Ok(Decision {
            status: "route",
            replica: Some(replica.replica.clone()),
            upstream: Some(replica.upstream.clone()),
            cached_tokens: c.cached_tokens,
            backlog_seconds: c.backlog_seconds,
            prefill_seconds: c.prefill_seconds,
            adapter_seconds: c.adapter_seconds,
            reason,
            retry_after: 0.0,
            available_replicas: available,
            best_cached_tokens: best_cached,
        })
    }

    /// Completion/cancellation always releases. Only real failures affect health.
    fn complete(&mut self, request_id: &str, outcome: &str) -> Result<(), String> {
        if !["ok", "busy", "failed", "cancelled"].contains(&outcome) {
            return Err(String::from("unknown routing outcome"));
        }
        if let Some(r) = self.reservations.remove(request_id) {
            let now = self.clock.elapsed().as_secs_f64();
            if outcome == "busy" {
                self.busy_until.insert(r.replica, now + 0.5);
            } else if outcome == "failed" {
                self.failed_until.insert(r.replica, now + 2.0);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn pending(&self) -> usize {
        self.reservations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn router(policy: &str, capacity: usize) -> Router {
        Router::new(policy, capacity, 1000., 100., 0.1, 1., 4, 1.25, 0.5).unwrap()
    }
    fn request(adapter: &str) -> Request {
        Request::new(adapter.into(), "problem".into(), vec![42; 1024], 100).unwrap()
    }
    fn replica(
        id: &str,
        active: usize,
        gpu: Vec<&str>,
        prefixes: Vec<String>,
    ) -> serde_json::Value {
        json!({"replica":id,"upstream":id,"active":active,"work":{
            "age_seconds":0.,"prefixes":prefixes,"gpu_adapters":gpu,"cpu_adapters":gpu,
            "pending_adapters":[],"request_ids":[],"prefill_tokens":0,"decode_tokens":0}})
    }
    #[test]
    fn reservations_bound_admission_and_busy_is_not_health_failure() {
        let mut r = router("cache_wait", 1);
        let q = request("m/v1");
        r.update_snapshot(&json!([replica("a", 0, vec![], vec![])]).to_string())
            .unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into(), 0.0).unwrap().status,
            "route"
        );
        assert_eq!(
            r.select_destination(&q, "2".into(), 0.0).unwrap().status,
            "wait"
        );
        r.complete("1", "busy").unwrap();
        assert_eq!(r.pending(), 0);
        assert!(r.failed_until.is_empty());
        assert_eq!(
            r.select_destination(&q, "2".into(), 0.0).unwrap().status,
            "wait"
        );
        r.busy_until.clear();
        assert_eq!(
            r.select_destination(&q, "2".into(), 2.01).unwrap().status,
            "route"
        );
        r.complete("2", "cancelled").unwrap();
        assert_eq!(r.pending(), 0);
    }
    #[test]
    fn snapshot_does_not_double_count_our_request() {
        let mut r = router("cache_wait", 2);
        let q = request("m/v1");
        let mut a = replica("a", 0, vec![], vec![]);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        r.select_destination(&q, "1".into(), 0.0).unwrap();
        a["active"] = json!(1);
        a["work"]["request_ids"] = json!(["1"]);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        assert_eq!(
            r.select_destination(&q, "2".into(), 2.01).unwrap().status,
            "route"
        );
    }
    #[test]
    fn stale_or_missing_observations_never_route() {
        let mut r = router("cache_wait", 1);
        let q = request("m/v1");
        assert_eq!(
            r.select_destination(&q, "1".into(), 0.0).unwrap().status,
            "unavailable"
        );
        let mut a = replica("a", 0, vec![], vec![]);
        a["work"]["age_seconds"] = json!(4.);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into(), 0.0).unwrap().status,
            "unavailable"
        );
    }
    #[test]
    fn adapter_versions_have_distinct_prefixes() {
        let a = request("m/v1");
        let b = request("m/v2");
        assert_ne!(a.hashes, b.hashes);
        assert_eq!(a.hashes.len() * 64, 960);
    }
    #[test]
    fn group_hints_are_scoped_and_released() {
        let mut r = router("cache_wait", 8);
        r.update_snapshot(
            &json!([
                replica("a", 0, vec![], vec![]),
                replica("b", 0, vec![], vec![])
            ])
            .to_string(),
        )
        .unwrap();
        let leader = r.prepare_group("group".into(), request("m/v1")).unwrap();
        let sibling = r.prepare_group("group".into(), request("m/v1")).unwrap();
        assert!(Arc::ptr_eq(&leader, &sibling));
        assert!(r.prepare_group("group".into(), request("m/v2")).is_err());
        let separate = r.prepare_group("other".into(), request("m/v1")).unwrap();
        assert!(!Arc::ptr_eq(&leader, &separate));
        r.select_destination(&leader, "leader".into(), 0.).unwrap();
        assert_eq!(
            r.select_destination(&sibling, "sibling".into(), 0.)
                .unwrap()
                .reason,
            "pending_prefix"
        );
        assert_eq!(
            r.select_destination(&sibling, "sibling".into(), 2.01)
                .unwrap()
                .status,
            "route"
        );
        r.complete("leader", "cancelled").unwrap();
        r.complete("sibling", "ok").unwrap();
        drop(leader);
        drop(sibling);
        drop(separate);
        r.prepare_group("next".into(), request("m/v1")).unwrap();
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.pending(), 0);
    }

    #[test]
    fn memory_guards_preserve_history_without_stranding_requests() {
        for (pinned, spare, hit, prompt, expected) in [
            (920_000, 700_000, 17152, 17334, "spare"),
            (850_000, 600_000, 17152, 17334, "cached"),
            (850_000, 650_000, 768, 884, "spare"),
            (920_000, 800_000, 17152, 17334, "cached"),
            (970_000, 950_000, 17152, 17334, "cached"),
            (700_000, 500_000, 768, 884, "cached"),
        ] {
            let mut r =
                Router::new("cache_wait", 136, 10463.96, 1290., 0.075, 0.68, 64, 2., 0.5).unwrap();
            let q = Request::new(
                "m/v1".into(),
                "conversation".into(),
                vec![42; prompt],
                16384,
            )
            .unwrap();
            let mut a = replica(
                "cached",
                125,
                vec!["m/v1"],
                vec![format!("{:032x}", q.hashes[hit / 64 - 1])],
            );
            let mut b = replica(
                "spare",
                100,
                vec!["m/v1"],
                vec![format!("{:032x}", q.hashes[5])],
            );
            for (row, used) in [(&mut a, pinned), (&mut b, spare)] {
                row["work"]["kv_used_tokens"] = json!(used);
                row["work"]["kv_capacity_tokens"] = json!(1_000_000);
            }
            r.update_snapshot(&json!([a, b]).to_string()).unwrap();
            assert_eq!(
                r.select_destination(&q, "1".into(), 0.)
                    .unwrap()
                    .replica
                    .as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn long_partial_prefix_outranks_load() {
        for (hit, expected) in [(1024, "cached"), (128, "spare")] {
            let mut r = router("cache_wait", 8);
            let q = Request::new("m/v1".into(), "problem".into(), vec![42; 4096], 100).unwrap();
            let a = replica(
                "cached",
                2,
                vec![],
                vec![format!("{:032x}", q.hashes[hit / 64 - 1])],
            );
            let b = replica("spare", 0, vec![], vec![]);
            r.update_snapshot(&json!([a, b]).to_string()).unwrap();
            assert_eq!(
                r.select_destination(&q, "1".into(), 0.)
                    .unwrap()
                    .replica
                    .as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn saturated_waits_end_with_the_affinity_budget() {
        let q = request("m/v1");
        let mut cached = replica("cached", 2, vec![], vec![format!("{:032x}", q.hashes[14])]);
        let mut grouped = replica("grouped", 2, vec![], vec![]);
        grouped["adapters"] = json!({"m/v1": 2});
        let mut spare = replica("spare", 0, vec![], vec![]);
        for row in [&mut cached, &mut grouped, &mut spare] {
            row["work"]["waiting_requests"] = json!(1);
        }
        for (target, reason) in [(cached, "cached_saturated"), (grouped, "adapter_saturated")] {
            let mut r = router("cache_wait", 2);
            r.update_snapshot(&json!([target, spare]).to_string())
                .unwrap();
            let waiting = r.select_destination(&q, "1".into(), 1.99).unwrap();
            assert_eq!((waiting.status, waiting.reason), ("wait", reason));
            let routed = r.select_destination(&q, "1".into(), 2.0).unwrap();
            assert_eq!(routed.replica.as_deref(), Some("spare"));
        }
    }

    #[test]
    fn kv_use_above_capacity_counts_as_full() {
        let mut r =
            Router::new("cache_wait", 136, 10463.96, 1290., 0.075, 0.68, 64, 2., 0.5).unwrap();
        let q = Request::new("m/v1".into(), "conversation".into(), vec![42; 884], 16384).unwrap();
        let mut a = replica(
            "cached",
            125,
            vec!["m/v1"],
            vec![format!("{:032x}", q.hashes[11])],
        );
        let mut b = replica("spare", 100, vec!["m/v1"], vec![]);
        a["work"]["kv_used_tokens"] = json!(1_000_001);
        a["work"]["kv_capacity_tokens"] = json!(1_000_000);
        b["work"]["kv_used_tokens"] = json!(500_000);
        b["work"]["kv_capacity_tokens"] = json!(1_000_000);
        r.update_snapshot(&json!([a, b]).to_string()).unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into(), 0.)
                .unwrap()
                .replica
                .as_deref(),
            Some("spare")
        );
    }

    #[test]
    fn snapshot_requires_an_upstream() {
        let mut r = router("cache_wait", 1);
        let mut a = replica("a", 0, vec![], vec![]);
        a.as_object_mut().unwrap().remove("upstream");
        assert!(r.update_snapshot(&json!([a]).to_string()).is_err());
    }

    #[test]
    fn stale_or_full_spare_cannot_force_migration() {
        for restriction in ["stale", "full"] {
            let mut r =
                Router::new("cache_wait", 136, 10463.96, 1290., 0.075, 0.68, 64, 2., 0.5).unwrap();
            let q = request("m/v1");
            let mut a = replica(
                "cached",
                125,
                vec![],
                vec![format!("{:032x}", q.hashes[14])],
            );
            let mut b = replica("spare", 100, vec![], vec![]);
            a["work"]["kv_used_tokens"] = json!(950_000);
            a["work"]["kv_capacity_tokens"] = json!(1_000_000);
            b["work"]["kv_used_tokens"] = json!(500_000);
            b["work"]["kv_capacity_tokens"] = json!(1_000_000);
            if restriction == "stale" {
                b["work"]["age_seconds"] = json!(4);
            } else {
                b["active"] = json!(136);
            }
            r.update_snapshot(&json!([a, b]).to_string()).unwrap();
            assert_eq!(
                r.select_destination(&q, "1".into(), 0.)
                    .unwrap()
                    .replica
                    .as_deref(),
                Some("cached")
            );
        }
    }
}
