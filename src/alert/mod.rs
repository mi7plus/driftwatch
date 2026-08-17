//! Pluggable alerting on drift detection.
//!
//! [`DatasetMonitor`](crate::DatasetMonitor) fires its configured [`Alerter`]
//! exactly when a check crosses the aggregate dataset-drift threshold. The
//! default is [`NopAlerter`], so a monitor is fully usable without wiring up any
//! alerting at all.
//!
//! Two implementations ship behind feature flags — [`LogAlerter`]
//! (`alerting-log`, emits a `tracing` event) and [`WebhookAlerter`]
//! (`alerting-webhook`, POSTs JSON). Any other destination (Slack, PagerDuty, a
//! database, …) is a matter of implementing [`Alerter`] yourself — it is a
//! single method, a few lines of code — rather than something this crate
//! special-cases per integration.

#[cfg(feature = "alerting-log")]
mod log_alerter;
#[cfg(feature = "alerting-webhook")]
mod webhook_alerter;

#[cfg(feature = "alerting-log")]
pub use log_alerter::LogAlerter;
#[cfg(feature = "alerting-webhook")]
pub use webhook_alerter::WebhookAlerter;

use crate::monitor::DriftReport;
use std::time::{SystemTime, UNIX_EPOCH};

/// One drifted feature's contribution to a [`DriftAlertEvent`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "alerting-webhook", derive(serde::Serialize))]
pub struct DriftedFeatureInfo {
    /// Name of the feature that drifted.
    pub feature: String,
    /// The primary metric that decided the verdict.
    pub metric: String,
    /// Observed statistic for that metric.
    pub score: f64,
    /// Threshold the statistic was compared against.
    pub threshold: f64,
}

/// The event handed to an [`Alerter`] when drift is detected.
///
/// Carries which features drifted, by which metric, the observed score vs.
/// threshold, and a timestamp (Unix seconds), so a handler has everything it
/// needs to render a message without reaching back into the monitor.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "alerting-webhook", derive(serde::Serialize))]
pub struct DriftAlertEvent {
    /// Unix timestamp (seconds) when the event was created.
    pub timestamp_secs: u64,
    /// Whether the aggregate dataset-drift verdict was positive.
    pub dataset_drifted: bool,
    /// Fraction of features that individually drifted, in `[0, 1]`.
    pub drifted_fraction: f64,
    /// Per-feature detail for every feature that drifted.
    pub features: Vec<DriftedFeatureInfo>,
}

impl DriftAlertEvent {
    /// Build an alert event summarizing the drifted features of a report.
    pub fn from_report(report: &DriftReport) -> Self {
        let features = report
            .drifted_features()
            .map(|f| DriftedFeatureInfo {
                feature: f.feature.clone(),
                metric: f.primary.to_string(),
                score: f.primary_statistic(),
                threshold: f.threshold,
            })
            .collect();
        Self {
            timestamp_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            dataset_drifted: report.dataset_drift_detected(),
            drifted_fraction: report.drifted_fraction(),
            features,
        }
    }
}

/// A sink for drift alerts.
///
/// Implement this for any destination not covered by the built-ins. Alerters
/// must be `Send + Sync` so a [`DatasetMonitor`](crate::DatasetMonitor) can be
/// shared across threads.
pub trait Alerter: Send + Sync {
    /// Handle a drift alert. Called by the monitor only when drift is detected.
    fn alert(&self, event: &DriftAlertEvent);
}

/// The default no-op alerter: drops every event. Lets a monitor run without any
/// alerting configured.
#[derive(Clone, Copy, Debug, Default)]
pub struct NopAlerter;

impl Alerter for NopAlerter {
    fn alert(&self, _event: &DriftAlertEvent) {}
}
