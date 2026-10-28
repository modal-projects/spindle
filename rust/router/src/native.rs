//! PyO3 hands each request to this runtime as one awaitable. Admission,
//! observations, retries and worker HTTP live in Rust.
use crate::{Admission, Decision, Outcome, Replica, Request, Router, Snapshot};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

const DELIVERIES_PER_ATTEMPT: u32 = 2;
const MAX_TRANSPORT_FAILURES: u32 = 8;
/// With nothing routable this long, Modal's gateway routes (and cold-starts)
/// instead, so a pool without fresh observations degrades rather than hangs.
const GATEWAY_FALLBACK: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Gateway stickiness headers; a pinned upstream replaces them.
const AFFINITY_HEADERS: &[&str] = &["modal-session-id", "modal-routing-affinity-key"];

#[derive(Deserialize)]
struct Config {
    gateway: String,
    headers: HashMap<String, String>,
    policy: String,
    capacity: usize,
    prefill_tokens_per_second: f64,
    balance_absolute: usize,
    balance_relative: f64,
    cache_threshold: f64,
}
#[derive(Clone, Deserialize)]
struct Worker {
    replica: String,
    upstream: String,
}
#[derive(Clone)]
struct Service {
    scheduler: Arc<Mutex<Router>>,
    workers: Arc<Mutex<Vec<Worker>>>,
    discovered: Arc<AtomicBool>,
    client: reqwest::Client,
    gateway: String,
    headers: HashMap<String, String>,
    policy: String,
    notify: Arc<Notify>,
    requests: Arc<Mutex<HashSet<String>>>,
    stop: CancellationToken,
}
#[derive(Deserialize)]
struct Generate {
    request_id: String,
    adapter: String,
    affinity: String,
    body: Value,
    headers: HashMap<String, String>,
    timeout_seconds: f64,
}
/// A worker's response, or the router's own status and message. Routing
/// metadata is present only when a worker was selected.
struct Routed {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
    replica: Option<String>,
    routing: Option<String>,
}
type Reply = Result<Routed, (u16, String)>;
fn error(status: u16, message: &str) -> (u16, String) {
    (status, message.into())
}
// RAII releases on success, timeout, cancellation, transport error and task abort.
struct Reservation {
    scheduler: Arc<Mutex<Router>>,
    id: String,
    outcome: Outcome,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.scheduler
            .lock()
            .unwrap()
            .complete(&self.id, self.outcome);
    }
}
struct ActiveRequest {
    service: Service,
    id: String,
}
impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.service.requests.lock().unwrap().remove(&self.id);
    }
}
async fn generate(s: Service, g: Generate) -> Reply {
    if !g.timeout_seconds.is_finite() || g.timeout_seconds <= 0.0 || g.timeout_seconds > 86400.0 {
        return Err(error(400, "Invalid deadline"));
    }
    if !s.requests.lock().unwrap().insert(g.request_id.clone()) {
        return Err(error(409, "Duplicate request ID"));
    }
    let _active = ActiveRequest {
        service: s.clone(),
        id: g.request_id.clone(),
    };
    tokio::select! {
        _=s.stop.cancelled()=>Err(error(503,"Router stopped")),
        result=tokio::time::timeout(Duration::from_secs_f64(g.timeout_seconds),forward(&s,g))=>
            result.unwrap_or_else(|_|Err(error(504,"Routing/dispatch deadline exceeded"))),
    }
}
async fn forward(s: &Service, mut g: Generate) -> Reply {
    let tokens: Vec<i64> =
        serde_json::from_value(g.body.get("input_ids").cloned().unwrap_or(Value::Null))
            .map_err(|_| error(400, "input_ids must be token IDs"))?;
    let budget = g
        .body
        .pointer("/sampling_params/max_new_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| error(400, "Missing output token budget"))? as usize;
    let request = Request::new(g.adapter, g.affinity, tokens, budget)
        .map_err(|_| error(400, "Invalid routing request"))?;
    let mut admission = Admission::default();
    let mut admission_seconds = 0.0;
    let mut attempt = 0;
    let mut failures: u32 = 0;
    let mut last: Option<Routed> = None;
    loop {
        let id = format!("{}:{attempt}", g.request_id);
        attempt += 1;
        let admission_started = Instant::now();
        let mut unavailable_since = None;
        let decision = loop {
            let d = s
                .scheduler
                .lock()
                .unwrap()
                .select_destination(&request, id.clone(), &mut admission)
                .map_err(|_| error(500, "Scheduler failed"))?;
            if d.status == "route" {
                break d;
            }
            if d.status == "unavailable" {
                let since = *unavailable_since.get_or_insert_with(Instant::now);
                let no_workers =
                    s.discovered.load(Ordering::Acquire) && s.workers.lock().unwrap().is_empty();
                if no_workers || since.elapsed() >= GATEWAY_FALLBACK {
                    break Decision::gateway();
                }
            } else {
                unavailable_since = None;
            }
            tokio::time::sleep(Duration::from_secs_f64(d.retry_after.clamp(0.01, 0.1))).await;
        };
        admission_seconds += admission_started.elapsed().as_secs_f64();
        let mut reservation = Reservation {
            scheduler: s.scheduler.clone(),
            id: id.clone(),
            outcome: Outcome::Released,
        };
        let route = json!({"policy":s.policy,"reason":decision.reason,"available_replicas":decision.available_replicas,
            "cached_tokens":decision.cached_tokens,"best_cached_tokens":decision.best_cached_tokens,
            "wait_s":admission_seconds});
        g.body["rid"] = json!(id);
        let mut headers = HeaderMap::new();
        for (k, v) in s.headers.iter().chain(g.headers.iter()) {
            let owned = k.eq_ignore_ascii_case("modal-flash-upstream")
                || (decision.upstream.is_some()
                    && AFFINITY_HEADERS.iter().any(|h| k.eq_ignore_ascii_case(h)));
            if owned {
                continue;
            }
            // Per-request values replace the runtime's, never duplicate them.
            let (Ok(name), Ok(value)) = (
                HeaderName::from_bytes(k.as_bytes()),
                HeaderValue::from_str(v),
            ) else {
                return Err(error(400, "Invalid worker request"));
            };
            headers.insert(name, value);
        }
        if let Some(upstream) = &decision.upstream {
            let value = HeaderValue::from_str(upstream)
                .map_err(|_| error(500, "Invalid worker upstream"))?;
            headers.insert("modal-flash-upstream", value);
        }
        let http = s
            .client
            .post(format!("{}/generate", s.gateway))
            .headers(headers)
            .json(&g.body)
            .build()
            .map_err(|_| error(400, "Invalid worker request"))?;
        let mut busy = false;
        // A lost connection may leave the generation running. Reconnect once
        // immediately with the same RID, payload and worker so its replay cache
        // can retain that work. Keep this reservation and the original deadline.
        for delivery in 0..DELIVERIES_PER_ATTEMPT {
            match s
                .client
                .execute(http.try_clone().expect("buffered JSON request"))
                .await
            {
                Ok(response) => {
                    let status = response.status().as_u16();
                    let response_headers: HashMap<String, String> = response
                        .headers()
                        .iter()
                        .filter_map(|(k, v)| {
                            v.to_str().ok().map(|v| (k.to_string(), v.to_string()))
                        })
                        .collect();
                    let actual = response_headers.get("x-spindle-replica").cloned();
                    if let (Some(actual), Some(expected)) = (&actual, &decision.replica) {
                        if actual != expected {
                            reservation.outcome = Outcome::Failed;
                            return Err(error(502, "Worker identity mismatch"));
                        }
                    }
                    let identified = actual.is_some();
                    let lost_upstream = !identified && (status == 408 || status >= 500);
                    let cancelled = status == 409
                        && identified
                        && matches!(
                            response_headers
                                .get("x-spindle-replay-state")
                                .map(String::as_str),
                            Some("cancelled" | "cancelling")
                        );
                    if lost_upstream && delivery == 0 {
                        continue;
                    }
                    busy = status == 429 || (status == 503 && identified);
                    // Any other identified status, 5xx included, is that worker's
                    // answer: return it rather than cool the replica down.
                    let retry = busy || cancelled || lost_upstream;
                    reservation.outcome = if busy {
                        Outcome::Busy
                    } else if lost_upstream {
                        Outcome::Failed
                    } else {
                        Outcome::Released
                    };
                    let bytes = match response.bytes().await {
                        Ok(bytes) => bytes,
                        Err(_) => {
                            reservation.outcome = Outcome::Failed;
                            if (200..300).contains(&status) && delivery == 0 {
                                continue;
                            }
                            if !retry && !(200..300).contains(&status) {
                                return Err(error(502, "Worker response interrupted"));
                            }
                            break;
                        }
                    };
                    let routed = Routed {
                        status,
                        headers: response_headers,
                        body: bytes.to_vec(),
                        replica: actual.or(decision.replica.clone()),
                        routing: Some(route.to_string()),
                    };
                    if !retry {
                        return Ok(routed);
                    }
                    last = Some(routed);
                }
                Err(err) => {
                    reservation.outcome = Outcome::Failed;
                    if delivery == 0 && !err.is_connect() {
                        continue;
                    }
                }
            }
            break;
        }
        drop(reservation);
        // Capacity rejection is admission pressure, bounded by generate's original
        // deadline. It must not consume the budget for transport/server failures.
        if !busy {
            failures += 1;
            if failures >= MAX_TRANSPORT_FAILURES {
                return last.ok_or_else(|| error(502, "Worker retry budget exhausted"));
            }
        }
        let delay_ms = if busy {
            100
        } else {
            100 * (1 << (failures - 1).min(4))
        };
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
}
async fn poll(s: Service, worker: Worker) -> Option<Replica> {
    let started = Instant::now();
    let mut req = s
        .client
        .get(format!("{}/routing-state", s.gateway))
        .header("Modal-Flash-Upstream", &worker.upstream)
        .timeout(Duration::from_secs(2));
    for (k, v) in &s.headers {
        req = req.header(k, v);
    }
    let response = req.send().await.ok()?.error_for_status().ok()?;
    let bytes = response.bytes().await.ok()?;
    let mut row: Replica = tokio::task::spawn_blocking(move || serde_json::from_slice(&bytes))
        .await
        .ok()?
        .ok()?;
    if row.replica != worker.replica {
        return None;
    }
    row.work.age_seconds += started.elapsed().as_secs_f64();
    row.upstream = worker.upstream;
    Some(row)
}
/// Each worker is polled on its own cadence and its row installed on arrival,
/// so a slow or unreachable worker never ages the others' observations.
async fn observe(s: Service) {
    let mut tasks = tokio::task::JoinSet::new();
    let mut polling: HashMap<tokio::task::Id, String> = HashMap::new();
    let mut tick = tokio::time::interval(POLL_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = s.stop.cancelled() => break,
            _ = s.notify.notified() => {}
            _ = tick.tick() => {}
            Some(done) = tasks.join_next_with_id(), if !tasks.is_empty() => {
                let (id, row) = match done {
                    Ok((id, row)) => (id, row),
                    Err(e) => (e.id(), None),
                };
                polling.remove(&id);
                if let Some(row) = row {
                    let snapshot = Snapshot::new(vec![row], Instant::now());
                    s.scheduler.lock().unwrap().install_snapshot(snapshot);
                }
                continue;
            }
        }
        let workers = s.workers.lock().unwrap().clone();
        for worker in workers {
            if polling
                .values()
                .any(|upstream| *upstream == worker.upstream)
            {
                continue;
            }
            let upstream = worker.upstream.clone();
            let handle = tasks.spawn(poll(s.clone(), worker));
            polling.insert(handle.id(), upstream);
        }
    }
}
#[pyclass]
pub struct NativeRuntime {
    service: Service,
}
#[pymethods]
impl NativeRuntime {
    #[new]
    fn new(config: &str) -> PyResult<Self> {
        let c: Config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let scheduler = Router::new(
            c.capacity,
            c.prefill_tokens_per_second,
            c.balance_absolute,
            c.balance_relative,
            c.cache_threshold,
        )
        .map_err(PyValueError::new_err)?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let service = Service {
            scheduler: Arc::new(Mutex::new(scheduler)),
            workers: Arc::new(Mutex::new(vec![])),
            discovered: Arc::new(AtomicBool::new(false)),
            client,
            gateway: c.gateway.trim_end_matches('/').into(),
            headers: c.headers,
            policy: c.policy,
            notify: Arc::new(Notify::new()),
            requests: Arc::new(Mutex::new(HashSet::new())),
            stop: CancellationToken::new(),
        };
        pyo3_async_runtimes::tokio::get_runtime().spawn(observe(service.clone()));
        Ok(Self { service })
    }

    /// Replace the discovered workers and observe them immediately.
    fn set_workers(&self, workers: &str) -> PyResult<()> {
        *self.service.workers.lock().unwrap() =
            serde_json::from_str(workers).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.service.discovered.store(true, Ordering::Release);
        self.service.notify.notify_one();
        Ok(())
    }

    /// Resolves to (status, headers, body, replica, routing JSON). Cancelling
    /// the awaitable drops the request and releases its reservation.
    fn generate<'py>(&self, py: Python<'py>, request: &str) -> PyResult<Bound<'py, PyAny>> {
        let g: Generate =
            serde_json::from_str(request).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let s = self.service.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let r = generate(s, g)
                .await
                .unwrap_or_else(|(status, message)| Routed {
                    status,
                    headers: HashMap::new(),
                    body: message.into_bytes(),
                    replica: None,
                    routing: None,
                });
            Ok((r.status, r.headers, r.body, r.replica, r.routing))
        })
    }

    fn health(&self) -> HashMap<&'static str, usize> {
        HashMap::from([
            (
                "pending",
                self.service.scheduler.lock().unwrap().reservations.len(),
            ),
            ("requests", self.service.requests.lock().unwrap().len()),
        ])
    }

    fn close(&self) {
        self.service.stop.cancel();
    }
}
impl Drop for NativeRuntime {
    fn drop(&mut self) {
        self.service.stop.cancel();
    }
}
