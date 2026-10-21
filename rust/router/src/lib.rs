//! One scheduler per frontend process and inference pool; native owns transport.
mod cache_wait;
mod native;
mod radix;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

#[derive(Deserialize)]
struct Work {
    age_seconds: f64,
    #[serde(default)]
    prefix_nodes: Vec<radix::Row>,
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
    upstream: String,
    active: usize,
    #[serde(default)]
    adapters: HashMap<String, usize>,
    work: Work,
}

#[pyclass(frozen)]
pub struct Request {
    adapter: String,
    affinity: String,
    prompt_tokens: usize,
    output_tokens: usize,
    hashes: Vec<u128>,
    preferred: Mutex<Option<String>>,
    pending_prefix: Mutex<Option<PendingPrefix>>,
}
struct PendingPrefix {
    replica: String,
    leader: String,
    started: Instant,
}

#[pymethods]
impl Request {
    #[new]
    fn new(
        adapter: String,
        affinity: String,
        tokens: Vec<i64>,
        output_tokens: usize,
    ) -> PyResult<Self> {
        if adapter.is_empty() || output_tokens == 0 {
            return Err(PyValueError::new_err(
                "adapter and output budget are required",
            ));
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
            hashes,
            preferred: Mutex::new(None),
            pending_prefix: Mutex::new(None),
        })
    }
}

#[pyclass(frozen, get_all)]
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
#[pyclass(frozen)]
pub struct Snapshot {
    replicas: Arc<Vec<Replica>>,
    radix: Arc<radix::Index>,
    observed: Instant,
}
impl Snapshot {
    fn build(json: &str) -> PyResult<Self> {
        let observed = Instant::now();
        let replicas: Vec<Replica> = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("invalid routing snapshot: {e}")))?;
        let mut ids = HashSet::new();
        if replicas.iter().any(|r| {
            !ids.insert(&r.replica) || !r.work.age_seconds.is_finite() || r.work.age_seconds < 0.0
        }) {
            return Err(PyValueError::new_err(
                "duplicate replica or invalid snapshot age",
            ));
        }
        let mut radix = radix::Index::default();
        for (i, r) in replicas.iter().enumerate() {
            radix
                .add(i, &r.work.prefix_nodes)
                .map_err(PyValueError::new_err)?;
        }
        Ok(Self {
            replicas: Arc::new(replicas),
            radix: Arc::new(radix),
            observed,
        })
    }
}
#[pymethods]
impl Snapshot {
    #[new]
    fn new(py: Python<'_>, json: String) -> PyResult<Self> {
        py.detach(move || Self::build(&json))
    }
}

#[pyclass]
pub struct Router {
    replicas: Arc<Vec<Replica>>,
    radix: Arc<radix::Index>,
    wait_policy: cache_wait::CacheWait,
    reservations: HashMap<String, Reservation>,
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
#[pymethods]
impl Router {
    #[new]
    #[allow(clippy::too_many_arguments)] // Flat Python configuration boundary.
    #[pyo3(signature = (policy, capacity, prefill_rate, decode_rate, gpu_load_seconds, registration_seconds, balance_absolute=4, balance_relative=1.25, cache_threshold=0.5))]
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
    ) -> PyResult<Self> {
        if policy != "cache_wait" {
            return Err(PyValueError::new_err("unknown routing policy"));
        }
        if !balance_relative.is_finite() || balance_relative < 1.0 {
            return Err(PyValueError::new_err(
                "balance_relative must be finite and >= 1",
            ));
        }
        if !cache_threshold.is_finite() || !(0.0..=1.0).contains(&cache_threshold) {
            return Err(PyValueError::new_err(
                "cache_threshold must be between 0 and 1",
            ));
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
            return Err(PyValueError::new_err("invalid capacity or service rates"));
        }
        Ok(Self {
            replicas: Arc::new(vec![]),
            radix: Arc::new(radix::Index::default()),
            wait_policy,
            reservations: HashMap::new(),
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

    /// Replace a complete observation, not an incremental request-path lookup.
    fn update_snapshot(&mut self, json: &str) -> PyResult<()> {
        let snapshot = Snapshot::build(json)?;
        self.install_snapshot(&snapshot);
        Ok(())
    }

    fn install_snapshot(&mut self, snapshot: &Snapshot) {
        self.replicas = snapshot.replicas.clone();
        self.radix = snapshot.radix.clone();
        self.updated =
            self.clock.elapsed().as_secs_f64() - snapshot.observed.elapsed().as_secs_f64();
    }

    /// Select and reserve in one short call. A busy pool stays busy, not unhealthy.
    #[pyo3(signature = (request, request_id, affinity_elapsed=0.0))]
    fn select_destination(
        &mut self,
        request: &Request,
        request_id: String,
        affinity_elapsed: f64,
    ) -> PyResult<Decision> {
        if !affinity_elapsed.is_finite() || affinity_elapsed < 0.0 {
            return Err(PyValueError::new_err("invalid affinity elapsed time"));
        }
        if self.reservations.contains_key(&request_id) {
            return Err(PyValueError::new_err("request already has a reservation"));
        }
        let now = self.clock.elapsed().as_secs_f64();
        let preferred = request.preferred.lock().unwrap().clone();
        let mut fresh = false;
        let mut candidates = Vec::new();
        let mut cooling_down = HashSet::new();
        let hits = self
            .radix
            .matches(&request.adapter, &request.hashes, self.replicas.len());
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
            // The structural catalog is bounded. Resident endpoints can still
            // prove longer matches when an ancestor was omitted from that catalog.
            let hit = request
                .hashes
                .iter()
                .enumerate()
                .rev()
                .take_while(|(index, _)| (index + 1) * 64 > hits[i])
                .find(|(_, hash)| w.prefixes.contains(hash))
                .map_or(hits[i], |(index, _)| (index + 1) * 64);
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
                kv_pinned_fraction: (w.kv_capacity_tokens > 0
                    && w.kv_used_tokens <= w.kv_capacity_tokens)
                    .then(|| w.kv_used_tokens as f64 / w.kv_capacity_tokens as f64),
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
    fn complete(&mut self, request_id: &str, outcome: &str) -> PyResult<()> {
        if !["ok", "busy", "failed", "cancelled"].contains(&outcome) {
            return Err(PyValueError::new_err("unknown routing outcome"));
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

    #[getter]
    fn pending(&self) -> usize {
        self.reservations.len()
    }
}

#[pymodule]
fn spindle_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<native::NativeRuntime>()?;
    m.add_class::<Router>()?;
    m.add_class::<Request>()?;
    m.add_class::<Snapshot>()?;
    m.add_class::<Decision>()?;
    Ok(())
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
}
