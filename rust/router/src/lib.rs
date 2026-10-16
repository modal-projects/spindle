//! One scheduler per frontend process and inference pool. No network I/O here.
mod policy;

use policy::{Joint, LeastLoad, Policy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

#[derive(Deserialize)]
struct Work {
    age_seconds: f64,
    prefixes: HashSet<String>,
    cpu_adapters: HashSet<String>,
    gpu_adapters: HashSet<String>,
    request_ids: HashSet<String>,
    pending_adapters: HashSet<String>,
    prefill_tokens: u64,
    decode_tokens: u64,
}
#[derive(Deserialize)]
struct Replica {
    replica: String,
    upstream: String,
    active: usize,
    work: Work,
}

#[pyclass(frozen)]
pub struct Request {
    adapter: String,
    affinity: String,
    prompt_tokens: usize,
    output_tokens: usize,
    prefixes: Vec<(String, usize)>,
    preferred: Mutex<Option<String>>,
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
        let mut prefixes = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            hasher.update(&token.to_le_bytes());
            // The final prompt token must be evaluated to obtain logits.
            if (i + 1) % 64 == 0 && i + 1 < tokens.len() {
                prefixes.push((format!("{:032x}", hasher.digest128()), i + 1));
            }
        }
        Ok(Self {
            adapter,
            affinity,
            prompt_tokens: tokens.len(),
            output_tokens,
            prefixes,
            preferred: Mutex::new(None),
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
        }
    }
}

pub struct Candidate {
    active: usize,
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

#[pyclass]
pub struct Router {
    policy: Box<dyn Policy>,
    replicas: Vec<Replica>,
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
    fn new(
        policy: &str,
        capacity: usize,
        prefill_rate: f64,
        decode_rate: f64,
        gpu_load_seconds: f64,
        registration_seconds: f64,
    ) -> PyResult<Self> {
        let policy: Box<dyn Policy> = match policy {
            "joint" => Box::new(Joint),
            "least_load" => Box::new(LeastLoad),
            _ => return Err(PyValueError::new_err("policy must be joint or least_load")),
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
            policy,
            replicas: vec![],
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
        self.replicas = replicas;
        self.updated = self.clock.elapsed().as_secs_f64();
        Ok(())
    }

    /// Select and reserve in one short call. A busy pool stays busy, not unhealthy.
    fn select_destination(&mut self, request: &Request, request_id: String) -> PyResult<Decision> {
        if self.reservations.contains_key(&request_id) {
            return Err(PyValueError::new_err("request already has a reservation"));
        }
        let now = self.clock.elapsed().as_secs_f64();
        let preferred = request.preferred.lock().unwrap().clone();
        let mut fresh = false;
        let mut chosen: Option<(usize, Candidate, f64, u64)> = None;
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
                || self
                    .busy_until
                    .get(&replica.replica)
                    .is_some_and(|t| *t > now)
            {
                continue;
            }
            let unobserved: Vec<_> = self
                .reservations
                .iter()
                .filter(|(id, r)| r.replica == replica.replica && !w.request_ids.contains(*id))
                .map(|(_, r)| r)
                .collect();
            let active = replica.active + unobserved.len();
            if active >= self.capacity {
                continue;
            }
            let hit = request
                .prefixes
                .iter()
                .rev()
                .find(|(hash, _)| w.prefixes.contains(hash))
                .map_or(0, |(_, n)| *n);
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
            let c = Candidate {
                active,
                cached_tokens: hit,
                backlog_seconds: backlog,
                prefill_seconds: (request.prompt_tokens - hit) as f64 / self.prefill_rate,
                adapter_seconds,
            };
            let score = self.policy.score(&c);
            let tie = xxh3_64(
                format!(
                    "{}\0{}\0{}",
                    request.adapter, request.affinity, replica.replica
                )
                .as_bytes(),
            );
            let keep_group = preferred.as_ref() == Some(&replica.replica);
            if keep_group
                || chosen.as_ref().is_none_or(|(_, _, best, rank)| {
                    score < *best || (score == *best && tie > *rank)
                })
            {
                chosen = Some((i, c, score, tie));
            }
            if keep_group {
                break;
            }
        }
        let Some((i, c, _, _)) = chosen else {
            return Ok(Decision::wait(if fresh { "wait" } else { "unavailable" }));
        };
        let replica = &self.replicas[i];
        *request.preferred.lock().unwrap() = Some(replica.replica.clone());
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
    m.add_class::<Router>()?;
    m.add_class::<Request>()?;
    m.add_class::<Decision>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn router(policy: &str, capacity: usize) -> Router {
        Router::new(policy, capacity, 1000., 100., 0.1, 1.).unwrap()
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
    fn joint_prices_kv_and_lora_together() {
        let mut r = router("joint", 2);
        let q = request("m/v1");
        // A hot adapter saves 1.1s, exceeding 0.96s of cache reuse on a cold host.
        r.update_snapshot(
            &json!([
                replica("hot", 0, vec!["m/v1"], vec![]),
                replica(
                    "cache",
                    0,
                    vec![],
                    q.prefixes.iter().map(|p| p.0.clone()).collect()
                )
            ])
            .to_string(),
        )
        .unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into())
                .unwrap()
                .replica
                .as_deref(),
            Some("hot")
        );
        let q = request("m/v1");
        r.update_snapshot(
            &json!([
                replica("hot", 0, vec!["m/v1"], vec![]),
                replica(
                    "cache",
                    0,
                    vec!["m/v1"],
                    q.prefixes.iter().map(|p| p.0.clone()).collect()
                )
            ])
            .to_string(),
        )
        .unwrap();
        assert_eq!(
            r.select_destination(&q, "2".into())
                .unwrap()
                .replica
                .as_deref(),
            Some("cache")
        );
    }
    #[test]
    fn reservations_bound_admission_and_busy_is_not_health_failure() {
        let mut r = router("joint", 1);
        let q = request("m/v1");
        r.update_snapshot(&json!([replica("a", 0, vec![], vec![])]).to_string())
            .unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into()).unwrap().status,
            "route"
        );
        assert_eq!(r.select_destination(&q, "2".into()).unwrap().status, "wait");
        r.complete("1", "busy").unwrap();
        assert_eq!(r.pending(), 0);
        assert!(r.failed_until.is_empty());
        assert_eq!(r.select_destination(&q, "2".into()).unwrap().status, "wait");
        r.busy_until.clear();
        assert_eq!(
            r.select_destination(&q, "2".into()).unwrap().status,
            "route"
        );
        r.complete("2", "cancelled").unwrap();
        assert_eq!(r.pending(), 0);
    }
    #[test]
    fn snapshot_does_not_double_count_our_request() {
        let mut r = router("least_load", 2);
        let q = request("m/v1");
        let mut a = replica("a", 0, vec![], vec![]);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        r.select_destination(&q, "1".into()).unwrap();
        a["active"] = json!(1);
        a["work"]["request_ids"] = json!(["1"]);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        assert_eq!(
            r.select_destination(&q, "2".into()).unwrap().status,
            "route"
        );
    }
    #[test]
    fn groups_reuse_a_destination_but_split_when_full() {
        let mut r = router("least_load", 2);
        let q = request("m/v1");
        r.update_snapshot(
            &json!([
                replica("a", 0, vec![], vec![]),
                replica("b", 0, vec![], vec![])
            ])
            .to_string(),
        )
        .unwrap();
        let one = r.select_destination(&q, "1".into()).unwrap();
        let two = r.select_destination(&q, "2".into()).unwrap();
        let three = r.select_destination(&q, "3".into()).unwrap();
        assert_eq!(one.replica, two.replica);
        assert_ne!(one.replica, three.replica);
    }
    #[test]
    fn stale_or_missing_observations_never_route() {
        let mut r = router("joint", 1);
        let q = request("m/v1");
        assert_eq!(
            r.select_destination(&q, "1".into()).unwrap().status,
            "unavailable"
        );
        let mut a = replica("a", 0, vec![], vec![]);
        a["work"]["age_seconds"] = json!(4.);
        r.update_snapshot(&json!([a]).to_string()).unwrap();
        assert_eq!(
            r.select_destination(&q, "1".into()).unwrap().status,
            "unavailable"
        );
    }
    #[test]
    fn adapter_versions_have_distinct_prefixes() {
        let a = request("m/v1");
        let b = request("m/v2");
        assert_ne!(a.prefixes, b.prefixes);
        assert_eq!(a.prefixes.last().unwrap().1, 960);
    }
}
