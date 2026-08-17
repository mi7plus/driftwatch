//! Webhook alerter test against a local mock HTTP server (no external endpoint).
#![cfg(feature = "alerting-webhook")]

use driftwatch::alert::{DriftAlertEvent, DriftedFeatureInfo};
use driftwatch::WebhookAlerter;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;

/// Spin up a one-shot mock HTTP server; return its URL and a receiver that
/// yields the request body once a request arrives.
fn mock_server() -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            // Split headers/body on the blank line.
            let body = request
                .split_once("\r\n\r\n")
                .map(|(_, b)| b.to_string())
                .unwrap_or_default();
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
            let _ = stream.write_all(response.as_bytes());
            let _ = tx.send(body);
        }
    });

    (format!("http://{addr}/hook"), rx)
}

fn sample_event() -> DriftAlertEvent {
    DriftAlertEvent {
        timestamp_secs: 1_700_000_000,
        dataset_drifted: true,
        drifted_fraction: 0.75,
        features: vec![DriftedFeatureInfo {
            feature: "score".into(),
            metric: "PSI".into(),
            score: 0.42,
            threshold: 0.25,
        }],
    }
}

#[test]
fn webhook_posts_json_body() {
    let (url, rx) = mock_server();
    let alerter = WebhookAlerter::new(url).unwrap();

    alerter.try_send(&sample_event()).unwrap();

    let body = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("server did not receive a request");
    assert!(
        body.contains("\"dataset_drifted\":true"),
        "body was: {body}"
    );
    assert!(body.contains("\"feature\":\"score\""), "body was: {body}");
    assert!(
        body.contains("\"drifted_fraction\":0.75"),
        "body was: {body}"
    );
}
