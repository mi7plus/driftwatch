//! The `DatasetMonitor` orchestration type.

use super::{DriftReport, DriftVerdict, FeatureDrift, MetricKind, MetricScore};
use crate::alert::{Alerter, DriftAlertEvent, NopAlerter};
use crate::distribution::{Comparison, FeatureKind, LiveFeature, ReferenceDistribution};
use crate::error::{DriftError, Result};
use crate::metrics::{chi_square_test, js_divergence, kl_divergence, ks_test, psi};

/// Default fraction of drifted features above which the whole dataset is flagged
/// as drifted (Evidently's convention).
pub const DEFAULT_DATASET_FRACTION_THRESHOLD: f64 = 0.5;

/// Default PSI threshold for a per-feature drift verdict (the "significant
/// change" band of the PSI convention).
pub const DEFAULT_PSI_THRESHOLD: f64 = 0.25;

/// Default significance level (alpha) for a hypothesis-test-driven verdict.
pub const DEFAULT_ALPHA: f64 = 0.05;

/// Per-feature metric configuration: which metrics to compute, which one drives
/// the verdict, and the threshold applied to it.
#[derive(Clone, Debug)]
pub struct FeatureConfig {
    /// The metric whose value decides the feature's verdict.
    pub primary: MetricKind,
    /// Threshold for the primary metric: an upper bound for divergence metrics
    /// (drift if `score > threshold`), or an alpha significance level for tests
    /// (drift if `p_value < threshold`).
    pub threshold: f64,
    /// Additional metrics to compute and report but not use for the verdict.
    pub additional: Vec<MetricKind>,
}

impl FeatureConfig {
    /// The default configuration for a feature of the given kind.
    ///
    /// Continuous: PSI (threshold [`DEFAULT_PSI_THRESHOLD`]) as primary, plus a
    /// KS test reported alongside. Categorical: PSI as primary, plus a
    /// chi-square test reported alongside.
    pub fn default_for(kind: FeatureKind) -> Self {
        match kind {
            FeatureKind::Continuous => FeatureConfig {
                primary: MetricKind::Psi,
                threshold: DEFAULT_PSI_THRESHOLD,
                additional: vec![MetricKind::Ks],
            },
            FeatureKind::Categorical => FeatureConfig {
                primary: MetricKind::Psi,
                threshold: DEFAULT_PSI_THRESHOLD,
                additional: vec![MetricKind::ChiSquare],
            },
        }
    }

    /// Every metric this config computes, primary first, without duplicates.
    fn metrics(&self) -> Vec<MetricKind> {
        let mut all = vec![self.primary];
        for &m in &self.additional {
            if !all.contains(&m) {
                all.push(m);
            }
        }
        all
    }
}

struct FeatureEntry {
    reference: ReferenceDistribution,
    config: FeatureConfig,
}

/// Holds one [`ReferenceDistribution`] per feature and turns a live batch into a
/// [`DriftReport`], firing an [`Alerter`] when the aggregate verdict crosses
/// threshold.
///
/// # Example
/// ```
/// use driftwatch::{DatasetMonitor, ReferenceDistribution, EqualFrequencyBinning, LiveFeature};
///
/// let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let reference =
///     ReferenceDistribution::fit_continuous("x", &baseline, EqualFrequencyBinning::default())
///         .unwrap();
///
/// let mut monitor = DatasetMonitor::new();
/// monitor.add_feature(reference);
///
/// let live: Vec<f64> = (50..150).map(|i| i as f64).collect(); // shifted up
/// let report = monitor.check(&[("x", LiveFeature::Continuous(&live))]).unwrap();
/// assert!(report.features[0].drifted());
/// ```
pub struct DatasetMonitor {
    features: Vec<FeatureEntry>,
    dataset_fraction_threshold: f64,
    min_live_samples: usize,
    alerter: Box<dyn Alerter>,
}

impl Default for DatasetMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetMonitor {
    /// Create an empty monitor with default thresholds and a no-op alerter.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            dataset_fraction_threshold: DEFAULT_DATASET_FRACTION_THRESHOLD,
            min_live_samples: 2,
            alerter: Box::new(NopAlerter),
        }
    }

    /// Add a feature with the default configuration for its kind.
    pub fn add_feature(&mut self, reference: ReferenceDistribution) -> &mut Self {
        let config = FeatureConfig::default_for(reference.kind());
        self.features.push(FeatureEntry { reference, config });
        self
    }

    /// Add a feature with an explicit metric configuration.
    pub fn add_feature_with_config(
        &mut self,
        reference: ReferenceDistribution,
        config: FeatureConfig,
    ) -> &mut Self {
        self.features.push(FeatureEntry { reference, config });
        self
    }

    /// Set the fraction-of-features threshold for the aggregate dataset verdict.
    pub fn with_dataset_fraction_threshold(mut self, fraction: f64) -> Self {
        self.dataset_fraction_threshold = fraction;
        self
    }

    /// Set the minimum number of live samples a feature must supply for a check;
    /// below this, [`check`](Self::check) returns [`DriftError::SampleTooSmall`]
    /// rather than a low-power result.
    pub fn with_min_live_samples(mut self, min: usize) -> Self {
        self.min_live_samples = min;
        self
    }

    /// Install an alerter, replacing the default [`NopAlerter`]. It fires only
    /// when a check's aggregate verdict is [`DriftVerdict::Drifted`].
    pub fn with_alerter(mut self, alerter: Box<dyn Alerter>) -> Self {
        self.alerter = alerter;
        self
    }

    /// Number of features registered.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether no features are registered.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Check a live batch against every registered feature and return a report.
    ///
    /// `live` pairs each feature name with its live values. Every registered
    /// feature must be present.
    ///
    /// # Errors
    /// * [`DriftError::UnknownFeature`] if a registered feature is missing from
    ///   `live` (or a live entry has the wrong kind).
    /// * [`DriftError::SampleTooSmall`] if a feature's live batch is below
    ///   [`with_min_live_samples`](Self::with_min_live_samples).
    /// * Any metric error (e.g. bin-count mismatch).
    pub fn check(&self, live: &[(&str, LiveFeature)]) -> Result<DriftReport> {
        let mut feature_reports = Vec::with_capacity(self.features.len());

        for entry in &self.features {
            let name = entry.reference.name();
            let live_feature = live
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, f)| *f)
                .ok_or_else(|| DriftError::UnknownFeature(name.to_string()))?;

            let n_samples = match live_feature {
                LiveFeature::Continuous(s) => s.len(),
                LiveFeature::Categorical(s) => s.len(),
            };
            if n_samples < self.min_live_samples {
                return Err(DriftError::SampleTooSmall {
                    kind: "drift check",
                    minimum: self.min_live_samples,
                    actual: n_samples,
                });
            }

            let comparison = entry.reference.compare(live_feature)?;
            let scores = self.compute_scores(&entry.config, &comparison)?;
            let verdict = feature_verdict(&entry.config, &scores);

            feature_reports.push(FeatureDrift {
                feature: name.to_string(),
                kind: comparison.kind,
                scores,
                primary: entry.config.primary,
                threshold: entry.config.threshold,
                verdict,
                reference_histogram: comparison.reference_hist,
                live_histogram: comparison.live_hist,
            });
        }

        let report = DriftReport {
            features: feature_reports,
            dataset_fraction_threshold: self.dataset_fraction_threshold,
        };

        #[cfg(feature = "prometheus-export")]
        crate::export::record_report(&report);

        if report.dataset_drift_detected() {
            self.alerter.alert(&DriftAlertEvent::from_report(&report));
        }

        Ok(report)
    }

    fn compute_scores(
        &self,
        config: &FeatureConfig,
        comparison: &Comparison,
    ) -> Result<Vec<MetricScore>> {
        let mut scores = Vec::new();
        for kind in config.metrics() {
            match compute_metric(kind, comparison) {
                Ok(score) => scores.push(score),
                // A *non-primary* metric that simply doesn't apply to this
                // feature (e.g. a chi-square on a single-category reference) is
                // dropped rather than failing the whole check. The primary
                // metric's errors always propagate.
                Err(DriftError::SampleTooSmall { .. }) if kind != config.primary => {}
                Err(e) => return Err(e),
            }
        }
        Ok(scores)
    }
}

fn compute_metric(kind: MetricKind, comparison: &Comparison) -> Result<MetricScore> {
    let (statistic, p_value) = match kind {
        MetricKind::Psi => (
            psi(&comparison.reference_hist, &comparison.live_hist)?,
            None,
        ),
        MetricKind::Kl => (
            kl_divergence(&comparison.reference_hist, &comparison.live_hist)?,
            None,
        ),
        MetricKind::Js => (
            js_divergence(&comparison.reference_hist, &comparison.live_hist)?,
            None,
        ),
        MetricKind::Ks => {
            let (reference, live) = comparison.raw_samples.ok_or_else(|| {
                DriftError::InvalidConfig(
                    "KS test requires continuous raw samples; not available for categorical features"
                        .into(),
                )
            })?;
            let r = ks_test(reference, live)?;
            (r.statistic, Some(r.p_value))
        }
        MetricKind::ChiSquare => {
            let r = chi_square_test(&comparison.reference_hist, &comparison.live_hist)?;
            (r.statistic, Some(r.p_value))
        }
    };
    Ok(MetricScore {
        kind,
        statistic,
        p_value,
    })
}

/// Decide a feature's verdict from its primary metric's score.
fn feature_verdict(config: &FeatureConfig, scores: &[MetricScore]) -> DriftVerdict {
    let primary = scores.iter().find(|s| s.kind == config.primary);
    let drifted = match primary {
        Some(score) => {
            if config.primary.higher_is_more_drift() {
                score.statistic > config.threshold
            } else {
                // Test metric: drift when the p-value is below alpha.
                score.p_value.map(|p| p < config.threshold).unwrap_or(false)
            }
        }
        None => false,
    };
    if drifted {
        DriftVerdict::Drifted
    } else {
        DriftVerdict::Stable
    }
}
