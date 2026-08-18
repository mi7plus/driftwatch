//! A live, auto-refreshing drift dashboard (feature `dashboard`).
//!
//! This is the "live monitoring UI" that the static-report path deliberately
//! isn't: a small HTTP server exposing a self-contained web page that polls a
//! JSON endpoint and re-renders the latest [`DriftReport`] every couple of
//! seconds. It is the piece Evidently's hosted UI provides in Python.
//!
//! It does **not** compute drift itself or own any scheduling — you keep calling
//! [`DatasetMonitor::check`](crate::DatasetMonitor::check) on whatever cadence
//! suits your service and hand each report to [`Dashboard::update`]. The
//! dashboard just makes the most recent result (and a short history) visible.
//!
//! # Two ways to serve it
//!
//! * Standalone — [`Dashboard::serve`] binds its own listener:
//!   ```no_run
//!   # async fn f() -> std::io::Result<()> {
//!   use driftwatch::Dashboard;
//!   let dashboard = Dashboard::new();
//!   // ... spawn a task that calls `dashboard.update(report)` periodically ...
//!   dashboard.serve("127.0.0.1:8080".parse().unwrap()).await
//!   # }
//!   ```
//! * Mounted into an existing `axum` app via [`Dashboard::router`], e.g.
//!   `Router::new().nest("/drift", dashboard.router())`, so drift monitoring
//!   lives on the same server as the model it watches.
//!
//! [`Dashboard`] is cheaply clonable (it is an [`Arc`] inside) — clone it to
//! share one dashboard between the updating task and the server.

mod page;

use crate::monitor::{DriftReport, DriftVerdict};
use axum::{extract::State, response::Html, routing::get, Json, Router};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// How many recent checks to retain for the dashboard's trend sparkline.
const MAX_HISTORY: usize = 200;

/// A shared, live view of the most recent drift check(s).
///
/// Clone it freely: every clone points at the same underlying state.
#[derive(Clone)]
pub struct Dashboard {
    inner: Arc<Inner>,
}

struct Inner {
    title: String,
    state: RwLock<State_>,
}

#[derive(Default)]
struct State_ {
    latest: Option<DriftReport>,
    updated_secs: u64,
    checks: u64,
    history: Vec<HistoryPoint>,
}

#[derive(Clone, Copy, serde::Serialize)]
struct HistoryPoint {
    t: u64,
    fraction: f64,
    drifted: bool,
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Dashboard {
    /// Create an empty dashboard with the default title.
    pub fn new() -> Self {
        Self::with_title("driftwatch")
    }

    /// Create an empty dashboard with a custom page title.
    pub fn with_title(title: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                title: title.into(),
                state: RwLock::new(State_::default()),
            }),
        }
    }

    /// Publish a fresh drift report as the dashboard's current state.
    ///
    /// Call this after each [`DatasetMonitor::check`](crate::DatasetMonitor::check).
    /// The report replaces whatever was shown before; its dataset-drift verdict
    /// is also appended to the trend history.
    pub fn update(&self, report: DriftReport) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let point = HistoryPoint {
            t: now,
            fraction: report.drifted_fraction(),
            drifted: report.dataset_drift_detected(),
        };
        let mut state = self.inner.state.write().expect("dashboard state poisoned");
        state.latest = Some(report);
        state.updated_secs = now;
        state.checks += 1;
        state.history.push(point);
        if state.history.len() > MAX_HISTORY {
            let overflow = state.history.len() - MAX_HISTORY;
            state.history.drain(0..overflow);
        }
    }

    /// Number of checks published so far.
    pub fn check_count(&self) -> u64 {
        self.inner
            .state
            .read()
            .expect("dashboard state poisoned")
            .checks
    }

    /// The `axum` router serving the dashboard: `GET /` (the web page) and
    /// `GET /api/report` (the JSON the page polls). Mount it anywhere, e.g.
    /// `Router::new().nest("/drift", dashboard.router())`.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/", get(index))
            .route("/api/report", get(api))
            .with_state(self.clone())
    }

    /// Bind `addr` and serve the dashboard until the future is dropped.
    ///
    /// # Errors
    /// Returns the underlying `std::io::Error` if binding or serving fails.
    pub async fn serve(&self, addr: SocketAddr) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, self.router()).await
    }

    fn api_response(&self) -> ApiResponse {
        let state = self.inner.state.read().expect("dashboard state poisoned");
        let features = state
            .latest
            .as_ref()
            .map(|report| {
                report
                    .features
                    .iter()
                    .map(|f| ApiFeature {
                        feature: f.feature.clone(),
                        kind: kind_label(f.kind),
                        verdict: verdict_label(f.verdict),
                        primary_metric: f.primary.label(),
                        primary_score: f.primary_statistic(),
                        threshold: f.threshold,
                        metrics: f
                            .scores
                            .iter()
                            .map(|s| ApiMetric {
                                metric: s.kind.label(),
                                statistic: s.statistic,
                                p_value: s.p_value,
                            })
                            .collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let (dataset_drifted, drifted_fraction, dataset_fraction_threshold) = state
            .latest
            .as_ref()
            .map(|r| {
                (
                    r.dataset_drift_detected(),
                    r.drifted_fraction(),
                    r.dataset_fraction_threshold,
                )
            })
            .unwrap_or((false, 0.0, 0.0));

        ApiResponse {
            title: self.inner.title.clone(),
            has_data: state.latest.is_some(),
            updated_secs: state.updated_secs,
            checks: state.checks,
            dataset_drifted,
            drifted_fraction,
            dataset_fraction_threshold,
            features,
            history: state.history.clone(),
        }
    }
}

/// The JSON the web page polls. A dedicated DTO (rather than serializing the
/// internal report types) keeps the wire shape stable and includes the computed
/// verdicts the page needs.
#[derive(serde::Serialize)]
struct ApiResponse {
    title: String,
    has_data: bool,
    updated_secs: u64,
    checks: u64,
    dataset_drifted: bool,
    drifted_fraction: f64,
    dataset_fraction_threshold: f64,
    features: Vec<ApiFeature>,
    history: Vec<HistoryPoint>,
}

#[derive(serde::Serialize)]
struct ApiFeature {
    feature: String,
    kind: &'static str,
    verdict: &'static str,
    primary_metric: &'static str,
    primary_score: f64,
    threshold: f64,
    metrics: Vec<ApiMetric>,
}

#[derive(serde::Serialize)]
struct ApiMetric {
    metric: &'static str,
    statistic: f64,
    p_value: Option<f64>,
}

fn kind_label(kind: crate::distribution::FeatureKind) -> &'static str {
    match kind {
        crate::distribution::FeatureKind::Continuous => "continuous",
        crate::distribution::FeatureKind::Categorical => "categorical",
    }
}

fn verdict_label(verdict: DriftVerdict) -> &'static str {
    match verdict {
        DriftVerdict::Stable => "stable",
        DriftVerdict::Drifted => "drifted",
    }
}

async fn index(State(dashboard): State<Dashboard>) -> Html<String> {
    Html(page::render(&dashboard.inner.title))
}

async fn api(State(dashboard): State<Dashboard>) -> Json<ApiResponse> {
    Json(dashboard.api_response())
}
