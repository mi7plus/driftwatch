//! Prometheus / `metrics` export (feature `prometheus-export`).
//!
//! This adds drift gauges to whatever `metrics` recorder your application has
//! already installed — it does **not** stand up its own exporter. In the guide's
//! deployment, that means the drift scores show up on the same `/metrics`
//! endpoint the `axum` server already exposes, alongside request latencies and
//! everything else.
//!
//! Two gauges are emitted per [`check`](crate::DatasetMonitor::check):
//!
//! * `driftwatch_feature_drift_score{feature, metric}` — each feature's primary
//!   drift statistic.
//! * `driftwatch_dataset_drift` — the aggregate verdict as `1.0` (drifted) or
//!   `0.0` (stable), so it is alertable via Prometheus rules independently of
//!   this crate's own [`Alerter`](crate::Alerter) mechanism.

mod prometheus_export;

pub use prometheus_export::record_report;

/// Gauge name for a feature's primary drift score.
pub const FEATURE_DRIFT_GAUGE: &str = "driftwatch_feature_drift_score";
/// Gauge name for the aggregate dataset-drift verdict (0.0 / 1.0).
pub const DATASET_DRIFT_GAUGE: &str = "driftwatch_dataset_drift";
