//! Orchestration: turning per-feature reference distributions into a
//! dataset-level [`DriftReport`], plus the [`LiveWindow`] buffer that feeds a
//! monitor from a running service.
//!
//! This is the layer a user actually calls day to day — [`DatasetMonitor`]
//! rather than the individual metric functions.

mod dataset_monitor;
mod live_window;
mod prediction_drift;

#[cfg(feature = "label-drift")]
mod label_drift;

pub use dataset_monitor::{
    DatasetMonitor, FeatureConfig, DEFAULT_ALPHA, DEFAULT_DATASET_FRACTION_THRESHOLD,
    DEFAULT_PSI_THRESHOLD,
};
pub use live_window::{LiveWindow, WindowMode};
pub use prediction_drift::PredictionDriftMonitor;

#[cfg(feature = "label-drift")]
pub use label_drift::{LabelDriftMonitor, LabelDriftReport};

use crate::distribution::FeatureKind;
use std::fmt;

/// Which drift metric a score came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetricKind {
    /// Population Stability Index (binned, magnitude only).
    Psi,
    /// KL divergence of live from reference (binned, magnitude only).
    Kl,
    /// Jensen–Shannon divergence (binned, symmetric, bounded).
    Js,
    /// Two-sample Kolmogorov–Smirnov test (raw continuous samples, has p-value).
    Ks,
    /// Chi-square homogeneity test (categorical, has p-value).
    ChiSquare,
}

impl MetricKind {
    /// Whether a *larger* value of this metric means *more* drift. True for the
    /// divergence metrics; false for the hypothesis tests, where the p-value is
    /// what matters and smaller means more drift.
    pub fn higher_is_more_drift(self) -> bool {
        matches!(self, MetricKind::Psi | MetricKind::Kl | MetricKind::Js)
    }

    /// Whether this metric is a hypothesis test carrying a p-value.
    pub fn is_test(self) -> bool {
        matches!(self, MetricKind::Ks | MetricKind::ChiSquare)
    }

    /// Short human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            MetricKind::Psi => "PSI",
            MetricKind::Kl => "KL",
            MetricKind::Js => "JS",
            MetricKind::Ks => "KS",
            MetricKind::ChiSquare => "Chi2",
        }
    }
}

impl fmt::Display for MetricKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One metric's result for one feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricScore {
    /// Which metric produced this score.
    pub kind: MetricKind,
    /// The raw statistic (PSI/KL/JS value, or KS/chi-square statistic).
    pub statistic: f64,
    /// The p-value, present only for the hypothesis tests.
    pub p_value: Option<f64>,
}

/// A drift verdict for a single feature or for the whole dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftVerdict {
    /// No drift detected against the configured threshold.
    Stable,
    /// Drift detected.
    Drifted,
}

impl DriftVerdict {
    /// Whether this verdict is [`DriftVerdict::Drifted`].
    pub fn is_drifted(self) -> bool {
        matches!(self, DriftVerdict::Drifted)
    }
}

impl fmt::Display for DriftVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DriftVerdict::Stable => "stable",
            DriftVerdict::Drifted => "DRIFTED",
        })
    }
}

/// Drift results for a single feature: every configured metric's score, and the
/// verdict driven by the feature's primary metric against its threshold.
#[derive(Clone, Debug)]
pub struct FeatureDrift {
    /// Feature name.
    pub feature: String,
    /// Whether the feature is continuous or categorical.
    pub kind: FeatureKind,
    /// Every configured metric's score for this feature.
    pub scores: Vec<MetricScore>,
    /// The metric whose value decides this feature's verdict.
    pub primary: MetricKind,
    /// The threshold applied to the primary metric (an upper bound for the
    /// divergence metrics, or an alpha significance level for the tests).
    pub threshold: f64,
    /// This feature's verdict.
    pub verdict: DriftVerdict,
}

impl FeatureDrift {
    /// The score for a given metric, if it was computed for this feature.
    pub fn score(&self, kind: MetricKind) -> Option<&MetricScore> {
        self.scores.iter().find(|s| s.kind == kind)
    }

    /// The primary metric's statistic.
    pub fn primary_statistic(&self) -> f64 {
        self.score(self.primary)
            .map(|s| s.statistic)
            .unwrap_or(f64::NAN)
    }

    /// Whether this feature drifted.
    pub fn drifted(&self) -> bool {
        self.verdict.is_drifted()
    }
}

/// A dataset-level drift report: per-feature results plus an aggregate verdict.
///
/// The aggregate follows Evidently's dataset-drift convention: the dataset is
/// flagged as drifted when the fraction of individually-drifted features exceeds
/// a configurable threshold (default 0.5).
#[derive(Clone, Debug)]
pub struct DriftReport {
    /// Per-feature drift results, in the order features were checked.
    pub features: Vec<FeatureDrift>,
    /// Fraction-of-features threshold above which the dataset is flagged.
    pub dataset_fraction_threshold: f64,
}

impl DriftReport {
    /// The features that individually drifted.
    pub fn drifted_features(&self) -> impl Iterator<Item = &FeatureDrift> {
        self.features.iter().filter(|f| f.drifted())
    }

    /// Fraction of features that individually drifted, in `[0, 1]`.
    pub fn drifted_fraction(&self) -> f64 {
        if self.features.is_empty() {
            return 0.0;
        }
        self.drifted_features().count() as f64 / self.features.len() as f64
    }

    /// Whether the aggregate dataset-drift verdict is [`DriftVerdict::Drifted`]:
    /// more than [`dataset_fraction_threshold`](Self::dataset_fraction_threshold)
    /// of features drifted.
    pub fn dataset_drift_detected(&self) -> bool {
        self.drifted_fraction() > self.dataset_fraction_threshold
    }

    /// The aggregate dataset verdict.
    pub fn dataset_verdict(&self) -> DriftVerdict {
        if self.dataset_drift_detected() {
            DriftVerdict::Drifted
        } else {
            DriftVerdict::Stable
        }
    }
}

impl fmt::Display for DriftReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DriftReport ({} features)", self.features.len())?;
        for feat in &self.features {
            write!(f, "  {:<20} {:>8} ", feat.feature, feat.verdict.to_string())?;
            let parts: Vec<String> = feat
                .scores
                .iter()
                .map(|s| match s.p_value {
                    Some(p) => format!("{}={:.4} (p={:.4})", s.kind, s.statistic, p),
                    None => format!("{}={:.4}", s.kind, s.statistic),
                })
                .collect();
            writeln!(f, "{}", parts.join(", "))?;
        }
        writeln!(
            f,
            "  dataset: {} ({:.0}% of features drifted, threshold {:.0}%)",
            self.dataset_verdict(),
            self.drifted_fraction() * 100.0,
            self.dataset_fraction_threshold * 100.0,
        )
    }
}
