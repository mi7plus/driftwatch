//! Prediction drift: feature drift applied to a model's output distribution.

use super::{DatasetMonitor, DriftReport};
use crate::binning::ContinuousBinning;
use crate::distribution::{LiveFeature, ReferenceDistribution};
use crate::error::Result;

/// The feature name under which the prediction is monitored.
pub const PREDICTION_FEATURE: &str = "prediction";

/// A convenience preset that monitors a model's output distribution for drift.
///
/// This is **not new mechanism** — prediction drift is "just" feature drift with
/// the model's prediction treated as a single monitored feature. This type
/// exists so that is a documented, named entry point rather than something every
/// user has to rediscover. Under the hood it holds an ordinary
/// [`DatasetMonitor`] with one feature named `"prediction"`; a check here is
/// identical to calling that monitor directly.
///
/// # Example
/// ```
/// use driftwatch::{PredictionDriftMonitor, EqualFrequencyBinning};
///
/// let reference: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
/// let mon = PredictionDriftMonitor::new(&reference, EqualFrequencyBinning::default()).unwrap();
///
/// let shifted: Vec<f64> = (0..100).map(|i| 0.5 + i as f64 / 100.0).collect();
/// let report = mon.check(&shifted).unwrap();
/// assert!(report.features[0].drifted());
/// ```
pub struct PredictionDriftMonitor {
    monitor: DatasetMonitor,
}

impl PredictionDriftMonitor {
    /// Fit a prediction-drift monitor from a reference set of model outputs.
    ///
    /// # Errors
    /// Propagates any error from fitting the underlying
    /// [`ReferenceDistribution`].
    pub fn new(reference_predictions: &[f64], binning: impl ContinuousBinning) -> Result<Self> {
        let reference = ReferenceDistribution::fit_continuous(
            PREDICTION_FEATURE,
            reference_predictions,
            binning,
        )?;
        let mut monitor = DatasetMonitor::new();
        monitor.add_feature(reference);
        Ok(Self { monitor })
    }

    /// Build from an already-configured [`DatasetMonitor`] (e.g. to attach an
    /// alerter or custom threshold). The monitor should have exactly the
    /// prediction feature registered.
    pub fn from_monitor(monitor: DatasetMonitor) -> Self {
        Self { monitor }
    }

    /// Mutable access to the underlying monitor, e.g. to install an alerter.
    pub fn monitor_mut(&mut self) -> &mut DatasetMonitor {
        &mut self.monitor
    }

    /// Check a batch of live predictions for drift against the reference outputs.
    ///
    /// # Errors
    /// Propagates any error from [`DatasetMonitor::check`].
    pub fn check(&self, live_predictions: &[f64]) -> Result<DriftReport> {
        self.monitor.check(&[(
            PREDICTION_FEATURE,
            LiveFeature::Continuous(live_predictions),
        )])
    }
}
