//! `tracing`-based alerter (feature `alerting-log`).

use super::{Alerter, DriftAlertEvent};

/// An [`Alerter`] that emits a structured `tracing` event on drift detection.
///
/// It deliberately reuses whatever `tracing` subscriber your application already
/// installs (the same setup the guide's Deployment/Monitoring chapters
/// establish), rather than introducing a separate logging story. The event is
/// emitted at `WARN` level on the `driftwatch` target with the drifted-feature
/// count and fraction as fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogAlerter;

impl LogAlerter {
    /// Create a new `LogAlerter`.
    pub fn new() -> Self {
        Self
    }
}

impl Alerter for LogAlerter {
    fn alert(&self, event: &DriftAlertEvent) {
        tracing::warn!(
            target: "driftwatch",
            dataset_drifted = event.dataset_drifted,
            drifted_fraction = event.drifted_fraction,
            drifted_features = event.features.len(),
            timestamp_secs = event.timestamp_secs,
            "data drift detected"
        );
        for f in &event.features {
            tracing::warn!(
                target: "driftwatch",
                feature = %f.feature,
                metric = %f.metric,
                score = f.score,
                threshold = f.threshold,
                "feature drift"
            );
        }
    }
}
