//! The live dashboard must serve the latest report as JSON and an HTML page.
#![cfg(feature = "dashboard")]

use driftwatch::{
    Dashboard, DatasetMonitor, EqualFrequencyBinning, LiveFeature, ReferenceDistribution,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

/// Minimal HTTP GET returning the response body.
async fn http_get(addr: &str, path: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf);
    text.split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default()
}

#[tokio::test]
async fn serves_latest_report_as_json_and_html() {
    // Build a monitor and produce a drifted report.
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "score",
            &linspace(0.0, 1.0, 500),
            EqualFrequencyBinning::new(10).unwrap(),
        )
        .unwrap(),
    );
    let shifted = linspace(2.0, 3.0, 500);
    let report = monitor
        .check(&[("score", LiveFeature::Continuous(&shifted))])
        .unwrap();
    assert!(report.dataset_drift_detected());

    // Publish it to the dashboard and serve on an ephemeral port.
    let dashboard = Dashboard::with_title("test-dash");
    dashboard.update(report);
    assert_eq!(dashboard.check_count(), 1);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let server = dashboard.clone();
    tokio::spawn(async move {
        axum::serve(listener, server.router()).await.unwrap();
    });

    // The JSON endpoint reflects the drifted report.
    let json = http_get(&addr, "/api/report").await;
    assert!(json.contains("\"dataset_drifted\":true"), "json: {json}");
    assert!(json.contains("\"feature\":\"score\""), "json: {json}");
    assert!(json.contains("\"title\":\"test-dash\""), "json: {json}");

    // The HTML page is self-contained (no external asset URLs) and titled.
    let html = http_get(&addr, "/").await;
    assert!(html.contains("test-dash"), "html missing title");
    assert!(html.contains("drift dashboard"));
    assert!(
        !html.contains("http://"),
        "page must not reference external assets"
    );
    assert!(
        !html.contains("https://"),
        "page must not reference external assets"
    );
}

#[tokio::test]
async fn empty_dashboard_reports_no_data() {
    let dashboard = Dashboard::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let server = dashboard.clone();
    tokio::spawn(async move {
        axum::serve(listener, server.router()).await.unwrap();
    });

    let json = http_get(&addr, "/api/report").await;
    assert!(json.contains("\"has_data\":false"), "json: {json}");
    assert!(json.contains("\"checks\":0"), "json: {json}");
}
