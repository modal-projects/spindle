//! PyO3 starts/stops this service only. Requests, observations and HTTP live in Rust.
use crate::{Request, Router, Snapshot};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{delete, get, post, put},
    Json,
};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
struct Config {
    gateway: String,
    token: String,
    headers: HashMap<String, String>,
    policy: String,
    capacity: usize,
    rates: [f64; 4],
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
    client: reqwest::Client,
    gateway: String,
    headers: HashMap<String, String>,
    token: String,
    policy: String,
    notify: Arc<Notify>,
    requests: Arc<Mutex<HashMap<String, CancellationToken>>>,
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
type Reply = Result<Json<Value>, (StatusCode, String)>;
fn error(status: StatusCode, message: &str) -> (StatusCode, String) {
    (status, message.into())
}
fn authorize(s: &Service, headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    if headers.get("authorization").and_then(|v| v.to_str().ok())
        != Some(&format!("Bearer {}", s.token))
    {
        return Err(error(
            StatusCode::UNAUTHORIZED,
            "Invalid local router token",
        ));
    }
    Ok(())
}
async fn workers(
    State(s): State<Service>,
    headers: HeaderMap,
    Json(rows): Json<Vec<Worker>>,
) -> Reply {
    authorize(&s, &headers)?;
    *s.workers.lock().unwrap() = rows;
    s.notify.notify_one();
    Ok(Json(json!({"updated":true})))
}
async fn health(State(s): State<Service>, headers: HeaderMap) -> Reply {
    authorize(&s, &headers)?;
    Ok(Json(
        json!({"pending":s.scheduler.lock().unwrap().reservations.len(),"requests":s.requests.lock().unwrap().len()}),
    ))
}
async fn cancel(State(s): State<Service>, headers: HeaderMap, Path(id): Path<String>) -> Reply {
    authorize(&s, &headers)?;
    if let Some(token) = s.requests.lock().unwrap().get(&id) {
        token.cancel();
    }
    Ok(Json(json!({"cancelled":true})))
}
// RAII releases on success, timeout, cancellation, transport error and task abort.
struct Reservation {
    scheduler: Arc<Mutex<Router>>,
    id: String,
    outcome: &'static str,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let _ = self
            .scheduler
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
async fn generate(State(s): State<Service>, headers: HeaderMap, Json(g): Json<Generate>) -> Reply {
    authorize(&s, &headers)?;
    if !g.timeout_seconds.is_finite() || g.timeout_seconds <= 0.0 || g.timeout_seconds > 86400.0 {
        return Err(error(StatusCode::BAD_REQUEST, "Invalid deadline"));
    }
    let token = CancellationToken::new();
    {
        let mut requests = s.requests.lock().unwrap();
        if requests.contains_key(&g.request_id) {
            return Err(error(StatusCode::CONFLICT, "Duplicate request ID"));
        }
        requests.insert(g.request_id.clone(), token.clone());
    }
    let _active = ActiveRequest {
        service: s.clone(),
        id: g.request_id.clone(),
    };
    tokio::select! {
        _=token.cancelled()=>Err(error(StatusCode::REQUEST_TIMEOUT,"Request cancelled")),
        _=s.stop.cancelled()=>Err(error(StatusCode::SERVICE_UNAVAILABLE,"Router stopped")),
        result=tokio::time::timeout(Duration::from_secs_f64(g.timeout_seconds),forward(&s,g))=>
            result.unwrap_or_else(|_|Err(error(StatusCode::GATEWAY_TIMEOUT,"Routing/dispatch deadline exceeded"))),
    }
}
async fn forward(s: &Service, mut g: Generate) -> Reply {
    let tokens: Vec<i64> =
        serde_json::from_value(g.body.get("input_ids").cloned().unwrap_or(Value::Null))
            .map_err(|_| error(StatusCode::BAD_REQUEST, "input_ids must be token IDs"))?;
    let budget = g
        .body
        .pointer("/sampling_params/max_new_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| error(StatusCode::BAD_REQUEST, "Missing output token budget"))?
        as usize;
    let request = Request::new(g.adapter, g.affinity, tokens, budget)
        .map_err(|_| error(StatusCode::BAD_REQUEST, "Invalid routing request"))?;
    let mut admission_seconds = 0.0;
    let mut affinity_started: Option<Instant> = None;
    let mut attempt = 0;
    let mut failures: u32 = 0;
    loop {
        let id = format!("{}:{attempt}", g.request_id);
        attempt += 1;
        let admission_started = Instant::now();
        let decision = loop {
            let elapsed = affinity_started.map_or(0.0, |t| t.elapsed().as_secs_f64());
            let d = s
                .scheduler
                .lock()
                .unwrap()
                .select_destination(&request, id.clone(), elapsed)
                .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "Scheduler failed"))?;
            if d.status == "route" {
                break d;
            }
            if matches!(d.reason, "pending_prefix" | "cached_capacity")
                && affinity_started.is_none()
            {
                affinity_started = Some(Instant::now());
            }
            tokio::time::sleep(Duration::from_secs_f64(d.retry_after.clamp(0.01, 0.1))).await;
        };
        admission_seconds += admission_started.elapsed().as_secs_f64();
        let mut reservation = Reservation {
            scheduler: s.scheduler.clone(),
            id: id.clone(),
            outcome: "cancelled",
        };
        let route = json!({"policy":s.policy,"reason":decision.reason,"available_replicas":decision.available_replicas,
            "cached_tokens":decision.cached_tokens,"best_cached_tokens":decision.best_cached_tokens,
            "backlog_seconds":decision.backlog_seconds,"prefill_seconds":decision.prefill_seconds,
            "adapter_seconds":decision.adapter_seconds,"wait_s":admission_seconds});
        g.body["rid"] = json!(id);
        let mut http = s
            .client
            .post(format!("{}/generate", s.gateway))
            .json(&g.body);
        for (k, v) in s.headers.iter().chain(g.headers.iter()) {
            if !k.eq_ignore_ascii_case("modal-session-id")
                && !k.eq_ignore_ascii_case("modal-routing-affinity-key")
                && !k.eq_ignore_ascii_case("modal-flash-upstream")
            {
                http = http.header(k, v);
            }
        }
        http = http.header(
            "Modal-Flash-Upstream",
            decision.upstream.as_deref().unwrap(),
        );
        let http = http
            .build()
            .map_err(|_| error(StatusCode::BAD_REQUEST, "Invalid worker request"))?;
        let mut busy = false;
        // A lost connection may leave the generation running. Reconnect once
        // immediately with the same RID, payload and worker so its replay cache
        // can retain that work. Keep this reservation and the original deadline.
        for delivery in 0..2 {
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
                    if let Some(actual) = response_headers.get("x-spindle-replica") {
                        if Some(actual) != decision.replica.as_ref() {
                            reservation.outcome = "failed";
                            return Err(error(StatusCode::BAD_GATEWAY, "Worker identity mismatch"));
                        }
                    }
                    let identified = response_headers.contains_key("x-spindle-replica");
                    let lost_upstream = !identified && (status == 408 || status >= 500);
                    let cancelled = status == 409
                        && identified
                        && matches!(
                            response_headers
                                .get("x-spindle-replay-state")
                                .map(String::as_str),
                            Some("cancelled" | "cancelling")
                        );
                    // Logical/version errors return to the caller; do not replay them blindly.
                    let retry = matches!(status, 408 | 429) || status >= 500 || cancelled;
                    if lost_upstream && delivery == 0 {
                        continue;
                    }
                    busy = status == 429 || (status == 503 && identified);
                    reservation.outcome = if busy {
                        "busy"
                    } else if retry {
                        "failed"
                    } else {
                        "ok"
                    };
                    let bytes = match response.bytes().await {
                        Ok(bytes) => bytes,
                        Err(_) => {
                            reservation.outcome = "failed";
                            if (200..300).contains(&status) && delivery == 0 {
                                continue;
                            }
                            if !retry && !(200..300).contains(&status) {
                                return Err(error(
                                    StatusCode::BAD_GATEWAY,
                                    "Worker response interrupted",
                                ));
                            }
                            break;
                        }
                    };
                    if !retry {
                        return Ok(Json(
                            json!({"status":status,"headers":response_headers,"body":String::from_utf8_lossy(&bytes),"routing":route,"replica":decision.replica,"attempts":attempt}),
                        ));
                    }
                }
                Err(err) => {
                    reservation.outcome = "failed";
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
            if failures >= 8 {
                return Err(error(
                    StatusCode::BAD_GATEWAY,
                    "Worker retry budget exhausted",
                ));
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
async fn observe(s: Service) {
    loop {
        let workers = s.workers.lock().unwrap().clone();
        let mut tasks = tokio::task::JoinSet::new();
        for worker in workers {
            let s = s.clone();
            tasks.spawn(async move {
                let started = Instant::now();
                let mut req = s
                    .client
                    .get(format!("{}/routing-state", s.gateway))
                    .query(&[("include_prefix_nodes", false)])
                    .header("Modal-Flash-Upstream", &worker.upstream)
                    .timeout(Duration::from_secs(2));
                for (k, v) in &s.headers {
                    req = req.header(k, v);
                }
                let response = req.send().await.ok()?.error_for_status().ok()?;
                let mut row: Value = response.json().await.ok()?;
                if row.get("replica")?.as_str()? != worker.replica {
                    return None;
                }
                let age =
                    row.pointer("/work/age_seconds")?.as_f64()? + started.elapsed().as_secs_f64();
                row["work"]["age_seconds"] = json!(age);
                row["upstream"] = json!(worker.upstream);
                Some(row)
            });
        }
        let mut rows = Vec::new();
        while let Some(result) = tasks.join_next().await {
            if let Ok(Some(row)) = result {
                rows.push(row);
            }
        }
        if let Ok(Ok(snapshot)) =
            tokio::task::spawn_blocking(move || Snapshot::build(&Value::Array(rows).to_string()))
                .await
        {
            s.scheduler.lock().unwrap().install_snapshot(&snapshot);
        }
        tokio::select! { _=s.stop.cancelled()=>break, _=s.notify.notified()=>{}, _=tokio::time::sleep(Duration::from_millis(500))=>{} }
    }
}
#[pyclass]
pub struct NativeRuntime {
    #[pyo3(get)]
    address: String,
    stop: CancellationToken,
    thread: Option<std::thread::JoinHandle<()>>,
}
#[pymethods]
impl NativeRuntime {
    #[new]
    fn new(config: String) -> PyResult<Self> {
        let c: Config = serde_json::from_str(&config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let scheduler = Router::new(
            &c.policy,
            c.capacity,
            c.rates[0],
            c.rates[1],
            c.rates[2],
            c.rates[3],
            c.balance_absolute,
            c.balance_relative,
            c.cache_threshold,
        )?;
        let stop = CancellationToken::new();
        let task_stop = stop.clone();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            let ready = tx.clone();
            let run = || -> Result<(), String> {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                        .await
                        .map_err(|e| e.to_string())?;
                    let address = format!(
                        "http://{}",
                        listener.local_addr().map_err(|e| e.to_string())?
                    );
                    let client = reqwest::Client::builder()
                        .no_proxy()
                        .redirect(reqwest::redirect::Policy::none())
                        .build()
                        .map_err(|e| e.to_string())?;
                    let s = Service {
                        scheduler: Arc::new(Mutex::new(scheduler)),
                        workers: Arc::new(Mutex::new(vec![])),
                        client,
                        gateway: c.gateway.trim_end_matches('/').into(),
                        headers: c.headers,
                        token: c.token,
                        policy: c.policy,
                        notify: Arc::new(Notify::new()),
                        requests: Arc::new(Mutex::new(HashMap::new())),
                        stop: task_stop.clone(),
                    };
                    tokio::spawn(observe(s.clone()));
                    let app = axum::Router::new()
                        .route("/generate", post(generate))
                        .route("/workers", put(workers))
                        .route("/health", get(health))
                        .route("/requests/{id}", delete(cancel))
                        .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024))
                        .with_state(s);
                    let _ = ready.send(Ok(address));
                    axum::serve(listener, app)
                        .with_graceful_shutdown(task_stop.cancelled_owned())
                        .await
                        .map_err(|e| e.to_string())
                })
            };
            if let Err(err) = run() {
                let _ = tx.send(Err(err));
            }
        });
        let address = rx
            .recv_timeout(Duration::from_secs(10))
            .map_err(|_| {
                stop.cancel();
                pyo3::exceptions::PyRuntimeError::new_err("Native router startup timed out")
            })?
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self {
            address,
            stop,
            thread: Some(thread),
        })
    }
    fn close(&mut self, py: Python<'_>) {
        self.stop.cancel();
        if let Some(thread) = self.thread.take() {
            py.detach(move || {
                let _ = thread.join();
            });
        }
    }
}
impl Drop for NativeRuntime {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}
