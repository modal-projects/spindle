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
/// Pools with no workers, or whose workers all publish no observations, use
/// the gateway at once.
const GATEWAY_FALLBACK: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Gateway stickiness headers; a pinned upstream replaces them.
const AFFINITY_HEADERS: &[&str] = &["modal-session-id", "modal-routing-affinity-key"];

#[derive(Deserialize)]
struct Config {
    gateway: Option<String>,
    headers: HashMap<String, String>,
    capacity: usize,
    prefill_tokens_per_second: Option<f64>,
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
    /// Upstreams whose latest poll answered without observations.
    unobservable: Arc<Mutex<HashSet<String>>>,
    measure_prefill: bool,
    client: reqwest::Client,
    gateway: Option<String>,
    headers: HashMap<String, String>,
    notify: Arc<Notify>,
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
/// SGLang reports these per request when metrics are enabled.
#[derive(Deserialize)]
struct Completion {
    meta_info: Timing,
}
#[derive(Deserialize)]
struct Timing {
    prompt_tokens: usize,
    #[serde(default)]
    cached_tokens: usize,
    forward_entry_time: f64,
    prefill_finished_time: f64,
    #[serde(default)]
    num_retractions: usize,
}
/// A retracted request restarts prefill, so its timestamps span two prefills.
fn prefill_sample(body: &[u8]) -> Option<(usize, f64)> {
    let t = serde_json::from_slice::<Completion>(body).ok()?.meta_info;
    (t.num_retractions == 0).then(|| {
        (
            t.prompt_tokens.saturating_sub(t.cached_tokens),
            t.prefill_finished_time - t.forward_entry_time,
        )
    })
}
enum Polled {
    Row(Box<Replica>),
    Unobservable,
}
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
async fn generate(s: Service, g: Generate) -> Reply {
    if !g.timeout_seconds.is_finite() || g.timeout_seconds <= 0.0 || g.timeout_seconds > 86400.0 {
        return Err(error(400, "Invalid deadline"));
    }
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
                let unobservable = s.discovered.load(Ordering::Acquire) && {
                    let workers = s.workers.lock().unwrap();
                    let unobservable = s.unobservable.lock().unwrap();
                    workers.iter().all(|w| unobservable.contains(&w.upstream))
                };
                if s.gateway.is_some() && (unobservable || since.elapsed() >= GATEWAY_FALLBACK) {
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
        let prefill_rate = s.scheduler.lock().unwrap().prefill_rate();
        let route = json!({"reason":decision.reason,"available_replicas":decision.available_replicas,
            "cached_tokens":decision.cached_tokens,"best_cached_tokens":decision.best_cached_tokens,
            "wait_s":admission_seconds,"prefill_tokens_per_second":prefill_rate});
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
        if let (Some(_), Some(upstream)) = (&s.gateway, &decision.upstream) {
            let value = HeaderValue::from_str(upstream)
                .map_err(|_| error(500, "Invalid worker upstream"))?;
            headers.insert("modal-flash-upstream", value);
        }
        let endpoint = s
            .gateway
            .as_ref()
            .or(decision.upstream.as_ref())
            .ok_or_else(|| error(503, "No worker available"))?;
        let http = s
            .client
            .post(format!("{}/generate", endpoint.trim_end_matches('/')))
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
                        if s.measure_prefill && (200..300).contains(&status) {
                            if let Some((tokens, seconds)) = prefill_sample(&routed.body) {
                                s.scheduler.lock().unwrap().record_prefill(tokens, seconds);
                            }
                        }
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
async fn poll(s: Service, worker: Worker) -> Option<Polled> {
    let started = Instant::now();
    let mut req = s
        .client
        .get(format!(
            "{}/routing-state",
            s.gateway
                .as_ref()
                .unwrap_or(&worker.upstream)
                .trim_end_matches('/')
        ))
        .timeout(Duration::from_secs(2));
    for (k, v) in &s.headers {
        if k.eq_ignore_ascii_case("modal-flash-upstream")
            || AFFINITY_HEADERS.iter().any(|h| k.eq_ignore_ascii_case(h))
        {
            continue;
        }
        req = req.header(k, v);
    }
    if s.gateway.is_some() {
        req = req.header("Modal-Flash-Upstream", &worker.upstream);
    }
    let response = req.send().await.ok()?.error_for_status().ok()?;
    let bytes = response.bytes().await.ok()?;
    let expected = worker.replica.clone();
    let polled = tokio::task::spawn_blocking(move || {
        let value: Value = serde_json::from_slice(&bytes).ok()?;
        if value.get("replica").and_then(Value::as_str) != Some(expected.as_str()) {
            return None;
        }
        // The worker answered but its cache publishes nothing to route by.
        if value.get("work").is_none_or(Value::is_null) {
            return Some(Polled::Unobservable);
        }
        serde_json::from_value(value)
            .ok()
            .map(|row| Polled::Row(Box::new(row)))
    })
    .await
    .ok()??;
    Some(match polled {
        Polled::Row(mut row) => {
            row.work.age_seconds += started.elapsed().as_secs_f64();
            row.upstream = worker.upstream;
            Polled::Row(row)
        }
        unobservable => unobservable,
    })
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
                let (id, polled) = match done {
                    Ok((id, polled)) => (id, polled),
                    Err(e) => (e.id(), None),
                };
                let Some(upstream) = polling.remove(&id) else { continue };
                match polled {
                    Some(Polled::Row(row)) => {
                        let workers = s.workers.lock().unwrap();
                        if !workers.iter().any(|w| w.upstream == row.upstream && w.replica == row.replica) {
                            continue;
                        }
                        s.unobservable.lock().unwrap().remove(&upstream);
                        let snapshot = Snapshot::new(vec![*row], Instant::now());
                        s.scheduler.lock().unwrap().install_snapshot(snapshot);
                    }
                    Some(Polled::Unobservable) => {
                        s.unobservable.lock().unwrap().insert(upstream);
                    }
                    None => {}
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
            unobservable: Arc::new(Mutex::new(HashSet::new())),
            measure_prefill: c.prefill_tokens_per_second.is_none(),
            client,
            gateway: c.gateway.map(|url| url.trim_end_matches('/').into()),
            headers: c.headers,
            notify: Arc::new(Notify::new()),
            stop: CancellationToken::new(),
        };
        pyo3_async_runtimes::tokio::get_runtime().spawn(observe(service.clone()));
        Ok(Self { service })
    }

    /// Replace the discovered workers and observe them immediately.
    fn set_workers(&self, workers: &str) -> PyResult<()> {
        let workers: Vec<Worker> =
            serde_json::from_str(workers).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if self.service.gateway.is_none() {
            for worker in &workers {
                let url = reqwest::Url::parse(&worker.upstream)
                    .map_err(|_| PyValueError::new_err("Direct workers require HTTP(S) URLs"))?;
                if !matches!(url.scheme(), "http" | "https")
                    || url.host_str().is_none()
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || url.query().is_some()
                    || url.fragment().is_some()
                {
                    return Err(PyValueError::new_err("Invalid direct worker URL"));
                }
            }
        }
        let mut current = self.service.workers.lock().unwrap();
        self.service
            .scheduler
            .lock()
            .unwrap()
            .replicas
            .retain(|row| {
                workers
                    .iter()
                    .any(|w| w.replica == row.replica && w.upstream == row.upstream)
            });
        *current = workers;
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

    /// Admission reservations still held.
    fn pending(&self) -> usize {
        self.service.scheduler.lock().unwrap().reservations.len()
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
