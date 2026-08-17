//! `axum` integration: a `LiveWindow` fed by a request handler, a periodic
//! `DatasetMonitor::check` driven by `tokio::time::interval`, and drift scores
//! exposed on the same `/metrics` endpoint the service already serves.
//!
//! This is the template the guide's Monitoring chapter adapts. It binds an
//! ephemeral port, drives itself with a burst of increasingly-drifted requests,
//! scrapes `/metrics`, prints it, and exits — so it runs cleanly as an example
//! rather than blocking forever.
//!
//! Run with:
//!
//! ```text
//! cargo run --example axum_integration --features prometheus-export
//! ```

use axum::{
    extract::{Query, State},
    routing::{get, post},
    Router,
};
use driftwatch::{
    DatasetMonitor, EqualFrequencyBinning, LiveFeature, LiveWindow, ReferenceDistribution,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone)]
struct AppState {
    window: Arc<LiveWindow>,
    monitor: Arc<DatasetMonitor>,
    prometheus: PrometheusHandle,
}

/// Mock inference handler: records the request's feature value into the live
/// window (in a real service this is where you'd log the model's input/output).
async fn predict(State(state): State<AppState>, Query(q): Query<HashMap<String, f64>>) -> String {
    let value = q.get("v").copied().unwrap_or(0.0);
    state.window.push(vec![value]);
    format!("scored {value}\n")
}

/// The metrics scrape endpoint — the same one the rest of the service exposes.
async fn metrics(State(state): State<AppState>) -> String {
    state.prometheus.render()
}

#[tokio::main]
async fn main() {
    // Install the Prometheus recorder the whole service (and driftwatch) reports to.
    let prometheus = PrometheusBuilder::new()
        .install_recorder()
        .expect("install prometheus recorder");

    // Fit the reference distribution for the monitored "score" feature.
    let baseline: Vec<f64> = (0..1000).map(|i| (i % 100) as f64 / 100.0).collect();
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(
        ReferenceDistribution::fit_continuous("score", &baseline, EqualFrequencyBinning::default())
            .unwrap(),
    );

    let state = AppState {
        window: Arc::new(LiveWindow::new(driftwatch::WindowMode::Count(500))),
        monitor: Arc::new(monitor),
        prometheus,
    };

    // The periodic drift check — the caller owns the cadence, not the crate.
    {
        let checker = state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(100));
            loop {
                ticker.tick().await;
                let live = checker.window.column(0);
                if live.len() >= 2 {
                    // Recording onto the Prometheus recorder happens inside check().
                    let _ = checker
                        .monitor
                        .check(&[("score", LiveFeature::Continuous(&live))]);
                }
            }
        });
    }

    // Start the server on an ephemeral port.
    let app = Router::new()
        .route("/predict", post(predict))
        .route("/metrics", get(metrics))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Drive the service with a burst of requests whose feature value drifts
    // upward over time, simulating a real distribution shift.
    for i in 0..400 {
        let v = 0.5 + i as f64 / 400.0; // 0.5 → 1.5, well above the [0,1) baseline
        http_post(&addr.to_string(), &format!("/predict?v={v}")).await;
    }

    // Give the interval checker a couple of ticks to run.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Scrape /metrics and show the drift gauges alongside everything else.
    let scrape = http_get(&addr.to_string(), "/metrics").await;
    println!("--- /metrics (driftwatch gauges) ---");
    for line in scrape.lines().filter(|l| l.contains("driftwatch")) {
        println!("{line}");
    }
}

/// Minimal HTTP/1.1 client so the example needs no HTTP client dependency.
async fn http_post(addr: &str, path: &str) -> String {
    request(addr, "POST", path).await
}
async fn http_get(addr: &str, path: &str) -> String {
    request(addr, "GET", path).await
}
async fn request(addr: &str, method: &str, path: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req =
        format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf);
    text.split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default()
}
