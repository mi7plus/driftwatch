//! The `metrics`-crate recording itself.

use super::{DATASET_DRIFT_GAUGE, FEATURE_DRIFT_GAUGE};
use crate::monitor::DriftReport;
use metrics::gauge;

/// Record a [`DriftReport`] onto the currently-installed `metrics` recorder.
///
/// Called automatically by [`DatasetMonitor::check`](crate::DatasetMonitor::check)
/// when the `prometheus-export` feature is enabled; also callable directly if
/// you build reports another way.
pub fn record_report(report: &DriftReport) {
    for feature in &report.features {
        gauge!(
            FEATURE_DRIFT_GAUGE,
            "feature" => feature.feature.clone(),
            "metric" => feature.primary.label(),
        )
        .set(feature.primary_statistic());
    }

    gauge!(DATASET_DRIFT_GAUGE).set(if report.dataset_drift_detected() {
        1.0
    } else {
        0.0
    });
}
