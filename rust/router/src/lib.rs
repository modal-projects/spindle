//! One scheduler per sampling process (`_execute_sample`'s container) and
//! inference pool. The policy does no I/O; its caller owns transport.
mod cache_wait;
mod native;

use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

/// Observations at least this old never route.
const MAX_OBSERVATION_AGE: f64 = 3.0;
/// Assumed until the pool's own prefills are measured.
const INITIAL_PREFILL_RATE: f64 = 10_000.0;
/// Shorter prefills measure scheduling overhead more than throughput.
const MIN_PREFILL_SAMPLE_TOKENS: usize = 1024;
/// Weight kept by earlier samples each time a prefill is measured.
const PREFILL_RATE_DECAY: f64 = 0.95;

/// Uncached prompt tokens per second of prefill, as configured or as measured
/// across the pool's own requests under their actual load.
enum PrefillRate {
    Fixed(f64),
    Measured { tokens: f64, seconds: f64 },
}
impl PrefillRate {
    fn get(&self) -> f64 {
        match *self {
            Self::Fixed(rate) => rate,
            Self::Measured { tokens, seconds } if seconds > 0.0 => tokens / seconds,
            Self::Measured { .. } => INITIAL_PREFILL_RATE,
        }
    }
    fn record(&mut self, uncached_tokens: usize, seconds: f64) {
        if let Self::Measured {
            tokens,
            seconds: total,
        } = self
        {
            if uncached_tokens >= MIN_PREFILL_SAMPLE_TOKENS && seconds.is_finite() && seconds > 0.0
            {
                *tokens = *tokens * PREFILL_RATE_DECAY + uncached_tokens as f64;
                *total = *total * PREFILL_RATE_DECAY + seconds;
            }
        }
    }
}

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
    hashes: Vec<u128>,
    preferred: Mutex<Option<String>>,
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
            hashes,
            preferred: Mutex::new(None),
        })
    }
}

/// One sample's admission state, kept by the caller across polls and attempts.
#[derive(Default)]
pub struct Admission {
    affinity_started: Option<Instant>,
}

pub struct Decision {
    pub status: &'static str,
    pub replica: Option<String>,
    pub upstream: Option<String>,
    pub cached_tokens: usize,
    pub reason: &'static str,
    pub retry_after: f64,
    pub available_replicas: usize,
    pub best_cached_tokens: usize,
}
impl Decision {
    /// Unpinned: Modal's gateway picks (or starts) the replica.
    fn gateway() -> Self {
        Self {
            status: "route",
            reason: "gateway_fallback",
            retry_after: 0.0,
            ..Self::wait("route")
        }
    }
    fn wait(status: &'static str) -> Self {
        Self {
            status,
            replica: None,
            upstream: None,
            cached_tokens: 0,
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
}
struct Reservation {
    replica: String,
    adapter: String,
}
/// How a reservation ended: completed or cancelled, rejected as busy, or failed.
#[derive(Clone, Copy)]
pub enum Outcome {
    Released,
    Busy,
    Failed,
}

/// Built away from the request path so installation holds the lock only to merge.
pub struct Snapshot {
    replicas: Vec<Replica>,
    observed: Instant,
}
impl Snapshot {
    /// Drops rows with a shared replica ID, no upstream or an invalid age.
    /// Those replicas keep their previous observation until it ages out.
    fn new(replicas: Vec<Replica>, observed: Instant) -> Self {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for r in &replicas {
            *counts.entry(r.replica.clone()).or_default() += 1;
        }
        let replicas = replicas
            .into_iter()
            .filter(|r| {
                counts[&r.replica] == 1
                    && !r.upstream.is_empty()
                    && r.work.age_seconds.is_finite()
                    && r.work.age_seconds >= 0.0
            })
            .collect();
        Self { replicas, observed }
    }
}
pub struct Router {
    replicas: Vec<Replica>,
    wait_policy: cache_wait::CacheWait,
    reservations: HashMap<String, Reservation>,
    busy_until: HashMap<String, f64>,
    failed_until: HashMap<String, f64>,
    clock: Instant,
    updated: f64,
    capacity: usize,
    prefill_rate: PrefillRate,
}
impl Router {
    /// Without a prefill rate, the router measures it from completed requests.
    fn new(
        capacity: usize,
        prefill_rate: Option<f64>,
        balance_absolute: usize,
        balance_relative: f64,
        cache_threshold: f64,
    ) -> Result<Self, String> {
        if !balance_relative.is_finite() || balance_relative < 1.0 {
            return Err(String::from("balance_relative must be finite and >= 1"));
        }
        if !cache_threshold.is_finite() || !(0.0..=1.0).contains(&cache_threshold) {
            return Err(String::from("cache_threshold must be between 0 and 1"));
        }
        if capacity == 0 || prefill_rate.is_some_and(|r| !r.is_finite() || r <= 0.0) {
            return Err(String::from("invalid capacity or prefill rate"));
        }
        let prefill_rate = match prefill_rate {
            Some(rate) => PrefillRate::Fixed(rate),
            None => PrefillRate::Measured {
                tokens: 0.0,
                seconds: 0.0,
            },
        };
        Ok(Self {
            replicas: vec![],
            wait_policy: cache_wait::CacheWait {
                absolute: balance_absolute,
                relative: balance_relative,
                threshold: cache_threshold,
            },
            reservations: HashMap::new(),
            busy_until: HashMap::new(),
            failed_until: HashMap::new(),
            clock: Instant::now(),
            updated: 0.0,
            capacity,
            prefill_rate,
        })
    }

    #[cfg(test)]
    fn update_snapshot(&mut self, json: &str) -> Result<(), String> {
        let observed = Instant::now();
        let rows = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
        self.install_snapshot(Snapshot::new(rows, observed));
        Ok(())
    }

    /// A replica missing from one poll keeps its last observation until it ages out.
    fn install_snapshot(&mut self, snapshot: Snapshot) {
        let updated =
            self.clock.elapsed().as_secs_f64() - snapshot.observed.elapsed().as_secs_f64();
        let observed: HashSet<String> = snapshot
            .replicas
            .iter()
            .map(|r| r.replica.clone())
            .collect();
        let mut replicas = snapshot.replicas;
        for mut previous in std::mem::take(&mut self.replicas) {
            previous.work.age_seconds += updated - self.updated;
            if !observed.contains(&previous.replica)
                && previous.work.age_seconds < MAX_OBSERVATION_AGE
            {
                replicas.push(previous);
            }
        }
        self.replicas = replicas;
        self.updated = updated;
    }

    /// Select and reserve in one short call. A busy pool stays busy, not unhealthy.
    fn select_destination(
        &mut self,
        request: &Request,
        request_id: String,
        admission: &mut Admission,
    ) -> Result<Decision, String> {
        if self.reservations.contains_key(&request_id) {
            return Err(String::from("request already has a reservation"));
        }
        let affinity_elapsed = admission
            .affinity_started
            .map_or(0.0, |t| t.elapsed().as_secs_f64());
        let now = self.clock.elapsed().as_secs_f64();
        let preferred = request.preferred.lock().unwrap().clone();
        let mut fresh = false;
        let mut candidates = Vec::new();
        let mut cooling_down = HashSet::new();
        for (i, replica) in self.replicas.iter().enumerate() {
            let w = &replica.work;
            if w.age_seconds + now - self.updated >= MAX_OBSERVATION_AGE {
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
            let adapter_pending = w.pending_adapters.contains(&request.adapter)
                || unobserved.iter().any(|r| r.adapter == request.adapter);
            candidates.push(Candidate {
                replica_index: i,
                adapter_tier: if w.gpu_adapters.contains(&request.adapter) {
                    0
                } else if adapter_pending {
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
            match policy.select(
                &all,
                &candidates,
                request,
                self.prefill_rate.get(),
                affinity_elapsed,
            ) {
                cache_wait::Selection::Wait(why, seconds) => {
                    admission.affinity_started.get_or_insert_with(Instant::now);
                    let mut d = Decision::wait("wait");
                    d.reason = why;
                    d.retry_after = seconds;
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
        self.reservations.insert(
            request_id,
            Reservation {
                replica: replica.replica.clone(),
                adapter: request.adapter.clone(),
            },
        );
        Ok(Decision {
            status: "route",
            replica: Some(replica.replica.clone()),
            upstream: Some(replica.upstream.clone()),
            cached_tokens: c.cached_tokens,
            reason,
            retry_after: 0.0,
            available_replicas: available,
            best_cached_tokens: best_cached,
        })
    }

    /// Every outcome releases the reservation. Only real failures affect health.
    fn complete(&mut self, request_id: &str, outcome: Outcome) {
        if let Some(r) = self.reservations.remove(request_id) {
            let now = self.clock.elapsed().as_secs_f64();
            match outcome {
                Outcome::Released => {}
                Outcome::Busy => {
                    self.busy_until.insert(r.replica, now + 0.5);
                }
                Outcome::Failed => {
                    self.failed_until.insert(r.replica, now + 2.0);
                }
            }
        }
    }

    fn record_prefill(&mut self, uncached_tokens: usize, seconds: f64) {
        self.prefill_rate.record(uncached_tokens, seconds);
    }

    fn prefill_rate(&self) -> f64 {
        self.prefill_rate.get()
    }

    #[cfg(test)]
    fn pending(&self) -> usize {
        self.reservations.len()
    }
}

#[pymodule]
fn spindle_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<native::NativeRuntime>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;
    fn router(capacity: usize) -> Router {
        Router::new(capacity, Some(1000.), 4, 1.25, 0.5).unwrap()
    }
    fn swe_router() -> Router {
        Router::new(136, Some(10463.96), 64, 2., 0.5).unwrap()
    }
    fn request(adapter: &str) -> Request {
        Request::new(adapter.into(), "problem".into(), vec![42; 1024], 100).unwrap()
    }
    /// An admission whose affinity clock started `seconds` ago.
    fn waited(seconds: f64) -> Admission {
        Admission {
            affinity_started: Some(Instant::now() - Duration::from_secs_f64(seconds)),
        }
    }
    fn fresh() -> Admission {
        Admission::default()
    }
    fn replica(
        id: &str,
        active: usize,
        gpu: Vec<&str>,
        prefixes: Vec<String>,
    ) -> serde_json::Value {
        json!({"replica":id,"upstream":id,"active":active,"work":{
            "age_seconds":0.,"prefixes":prefixes,"gpu_adapters":gpu,"cpu_adapters":gpu,
            "pending_adapters":[],"request_ids":[]}})
    }
    #[test]
    fn reservations_bound_admission_and_busy_is_not_health_failure() {
        let mut r = router(1);
        let q = request("m/v1");
        r.update_snapshot(&json!([replica("a", 0, vec![], vec![])]).to_string())
            .unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into(), &mut fresh())
                .unwrap()
                .status,
            "route"
        );
        assert_eq!(
            r.select_destination(&q, "2".into(), &mut fresh())
                .unwrap()
                .status,
            "wait"
        );
        r.complete("1", Outcome::Busy);
        assert_eq!(r.pending(), 0);
        assert!(r.failed_until.is_empty());
        assert_eq!(
            r.select_destination(&q, "2".into(), &mut fresh())
                .unwrap()
                .status,
            "wait"
        );
        r.busy_until.clear();
        assert_eq!(
            r.select_destination(&q, "2".into(), &mut waited(2.01))
                .unwrap()
                .status,
            "route"
        );
        r.complete("2", Outcome::Released);
        assert_eq!(r.pending(), 0);
    }
    #[test]
    fn snapshot_does_not_double_count_our_request() {
        let mut r = router(2);
        let q = request("m/v1");
        let mut a = replica("a", 0, vec![], vec![]);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        r.select_destination(&q, "1".into(), &mut fresh()).unwrap();
        a["active"] = json!(1);
        a["work"]["request_ids"] = json!(["1"]);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        assert_eq!(
            r.select_destination(&q, "2".into(), &mut waited(2.01))
                .unwrap()
                .status,
            "route"
        );
    }
    #[test]
    fn snapshot_intake() {
        for case in ["none", "stale", "missed_poll", "invalid_row"] {
            let capacity = if case == "missed_poll" || case == "invalid_row" {
                8
            } else {
                1
            };
            let mut r = router(capacity);
            let q = request("m/v1");
            match case {
                "none" => assert_eq!(
                    r.select_destination(&q, "1".into(), &mut fresh())
                        .unwrap()
                        .status,
                    "unavailable"
                ),
                "stale" => {
                    let mut a = replica("a", 0, vec![], vec![]);
                    a["work"]["age_seconds"] = json!(4.);
                    r.update_snapshot(&json!([a]).to_string()).unwrap();
                    assert_eq!(
                        r.select_destination(&q, "1".into(), &mut fresh())
                            .unwrap()
                            .status,
                        "unavailable"
                    );
                }
                "missed_poll" => {
                    let mut a = replica("a", 0, vec![], vec![]);
                    a["work"]["age_seconds"] = json!(2.5);
                    r.update_snapshot(&json!([a, replica("b", 4, vec![], vec![])]).to_string())
                        .unwrap();
                    r.update_snapshot(&json!([replica("b", 4, vec![], vec![])]).to_string())
                        .unwrap();
                    assert_eq!(
                        r.select_destination(&q, "1".into(), &mut fresh())
                            .unwrap()
                            .replica
                            .as_deref(),
                        Some("a")
                    );
                    r.updated -= 0.5;
                    r.update_snapshot(&json!([replica("b", 4, vec![], vec![])]).to_string())
                        .unwrap();
                    assert_eq!(
                        r.select_destination(&q, "2".into(), &mut fresh())
                            .unwrap()
                            .replica
                            .as_deref(),
                        Some("b")
                    );
                }
                "invalid_row" => {
                    let mut negative = replica("negative", 0, vec![], vec![]);
                    negative["work"]["age_seconds"] = json!(-1.);
                    let mut no_upstream = replica("no-upstream", 0, vec![], vec![]);
                    no_upstream.as_object_mut().unwrap().remove("upstream");
                    r.update_snapshot(
                        &json!([
                            negative,
                            no_upstream,
                            replica("twice", 0, vec![], vec![]),
                            replica("twice", 0, vec![], vec![]),
                            replica("valid", 3, vec![], vec![]),
                        ])
                        .to_string(),
                    )
                    .unwrap();
                    // Any surviving idle row would outrank the loaded valid replica.
                    assert_eq!(
                        r.select_destination(&q, "1".into(), &mut fresh())
                            .unwrap()
                            .replica
                            .as_deref(),
                        Some("valid")
                    );
                }
                _ => unreachable!(),
            }
        }
    }
    #[test]
    fn adapter_versions_have_distinct_prefixes() {
        let a = request("m/v1");
        let b = request("m/v2");
        assert_ne!(a.hashes, b.hashes);
        assert_eq!(a.hashes.len() * 64, 960);
    }
    #[test]
    fn memory_guards() {
        // pinned, spare used, spare active, spare age, hit, prompt, output, gpu, short spare, expected
        for (
            pinned,
            spare_used,
            spare_active,
            spare_age,
            hit,
            prompt,
            output,
            gpu,
            short_spare,
            expected,
        ) in [
            (
                920_000, 700_000, 100, 0., 17152, 17334, 16384, true, true, "spare",
            ),
            (
                850_000, 600_000, 100, 0., 17152, 17334, 16384, true, true, "cached",
            ),
            (
                850_000, 650_000, 100, 0., 768, 884, 16384, true, true, "spare",
            ),
            (
                920_000, 800_000, 100, 0., 17152, 17334, 16384, true, true, "cached",
            ),
            (
                970_000, 950_000, 100, 0., 17152, 17334, 16384, true, true, "cached",
            ),
            (
                700_000, 500_000, 100, 0., 768, 884, 16384, true, true, "cached",
            ),
            (
                1_000_001, 500_000, 100, 0., 768, 884, 16384, true, false, "spare",
            ),
            (
                950_000, 500_000, 100, 4., 960, 1024, 100, false, false, "cached",
            ),
            (
                950_000, 500_000, 136, 0., 960, 1024, 100, false, false, "cached",
            ),
        ] {
            let mut r = swe_router();
            let q = Request::new(
                "m/v1".into(),
                "conversation".into(),
                vec![42; prompt],
                output,
            )
            .unwrap();
            let adapters: Vec<&str> = if gpu { vec!["m/v1"] } else { vec![] };
            let mut a = replica(
                "cached",
                125,
                adapters.clone(),
                vec![format!("{:032x}", q.hashes[hit / 64 - 1])],
            );
            let spare_prefix = if short_spare {
                vec![format!("{:032x}", q.hashes[5])]
            } else {
                vec![]
            };
            let mut b = replica("spare", spare_active, adapters, spare_prefix);
            for (row, used) in [(&mut a, pinned), (&mut b, spare_used)] {
                row["work"]["kv_used_tokens"] = json!(used);
                row["work"]["kv_capacity_tokens"] = json!(1_000_000);
            }
            b["work"]["age_seconds"] = json!(spare_age);
            r.update_snapshot(&json!([a, b]).to_string()).unwrap();
            assert_eq!(
                r.select_destination(&q, "1".into(), &mut fresh())
                    .unwrap()
                    .replica
                    .as_deref(),
                Some(expected),
                "pinned={pinned} spare={spare_used} hit={hit}"
            );
        }
    }

    #[test]
    fn long_partial_prefix_outranks_load() {
        for (hit, expected) in [(1024, "cached"), (128, "spare")] {
            let mut r = router(8);
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
                r.select_destination(&q, "1".into(), &mut fresh())
                    .unwrap()
                    .replica
                    .as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn affinity_budget() {
        for kind in ["cached_saturated", "capacity"] {
            let q = request("m/v1");
            let mut r = router(if kind == "capacity" { 1 } else { 2 });
            if kind == "capacity" {
                r.update_snapshot(&json!([replica("a", 1, vec![], vec![])]).to_string())
                    .unwrap();
            } else {
                let mut cached =
                    replica("cached", 2, vec![], vec![format!("{:032x}", q.hashes[14])]);
                let mut spare = replica("spare", 0, vec![], vec![]);
                for row in [&mut cached, &mut spare] {
                    row["work"]["waiting_requests"] = json!(1);
                }
                r.update_snapshot(&json!([cached, spare]).to_string())
                    .unwrap();
            }
            let mut admission = fresh();
            let waiting = r
                .select_destination(&q, "1".into(), &mut admission)
                .unwrap();
            if kind == "capacity" {
                assert_eq!((waiting.status, waiting.reason), ("wait", "capacity"));
                assert!(admission.affinity_started.is_none());
                continue;
            }
            assert_eq!(
                (waiting.status, waiting.reason),
                ("wait", "cached_saturated")
            );
            assert!(admission.affinity_started.is_some());
            admission.affinity_started = Some(Instant::now() - Duration::from_secs(2));
            assert_eq!(
                r.select_destination(&q, "1".into(), &mut admission)
                    .unwrap()
                    .replica
                    .as_deref(),
                Some("spare")
            );
        }
    }
}
