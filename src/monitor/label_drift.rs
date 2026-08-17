//! Label / concept drift via a rolling scoring metric (feature `label-drift`).

use crate::error::{DriftError, Result};
use model_selection_rs::scoring::Scorer;
use ndarray::Array1;

/// Result of a label-drift check: the rolling score over recent
/// (prediction, actual) pairs versus the reference baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelDriftReport {
    /// Score over the current window of labeled pairs.
    pub rolling_score: f64,
    /// The reference/baseline score to compare against.
    pub baseline_score: f64,
    /// Signed degradation: how much *worse* the rolling score is than baseline,
    /// oriented by the scorer's `greater_is_better`. Positive means degraded.
    pub degradation: f64,
    /// Whether the degradation exceeded the configured threshold.
    pub drifted: bool,
}

/// Tracks model quality online by scoring a rolling window of
/// (prediction, eventual-actual-label) pairs and comparing against a baseline.
///
/// This is genuine **label/concept drift** — the relationship between inputs and
/// the target has shifted enough to hurt live accuracy — as opposed to the
/// input-distribution drift the other monitors detect. It reuses
/// `model-selection-rs`'s [`Scorer`] trait deliberately, so the metric measured
/// online here is the *same* abstraction the guide's Evaluation chapter measures
/// offline, rather than a separate metric definition.
///
/// # Delayed ground truth
///
/// This requires the true label to eventually arrive. In production, predictions
/// are made now but actuals land later (a purchase is or isn't made, a claim is
/// or isn't fraud). The realistic pattern is a queue that pairs each prediction
/// with its label once known, then periodically feeds the accumulated pairs to
/// [`check`](LabelDriftMonitor::check). Labels are *not* assumed to be
/// immediately available — that would defeat the purpose of a live monitor.
pub struct LabelDriftMonitor<S: Scorer> {
    scorer: S,
    baseline_score: f64,
    degradation_threshold: f64,
}

impl<S: Scorer> LabelDriftMonitor<S> {
    /// Create a monitor with an explicit baseline score.
    ///
    /// `degradation_threshold` is the absolute allowed drop in the scorer's
    /// value before drift is flagged (interpreted in the "worse" direction per
    /// the scorer's `greater_is_better`).
    pub fn new(scorer: S, baseline_score: f64, degradation_threshold: f64) -> Self {
        Self {
            scorer,
            baseline_score,
            degradation_threshold,
        }
    }

    /// Create a monitor whose baseline is the scorer's value over a reference set
    /// of (prediction, actual) pairs — e.g. the held-out test set the model was
    /// evaluated on offline.
    ///
    /// # Errors
    /// [`DriftError::EmptyInput`] if `reference_pairs` is empty.
    pub fn from_reference(
        scorer: S,
        reference_pairs: &[(f64, f64)],
        degradation_threshold: f64,
    ) -> Result<Self> {
        let baseline_score = score_pairs(&scorer, reference_pairs)?;
        Ok(Self::new(scorer, baseline_score, degradation_threshold))
    }

    /// The reference baseline score.
    pub fn baseline_score(&self) -> f64 {
        self.baseline_score
    }

    /// Score a window of (prediction, actual) pairs and compare to baseline.
    ///
    /// # Errors
    /// [`DriftError::EmptyInput`] if `pairs` is empty.
    pub fn check(&self, pairs: &[(f64, f64)]) -> Result<LabelDriftReport> {
        let rolling_score = score_pairs(&self.scorer, pairs)?;
        // Orient "degradation" so positive always means worse.
        let degradation = if self.scorer.greater_is_better() {
            self.baseline_score - rolling_score
        } else {
            rolling_score - self.baseline_score
        };
        Ok(LabelDriftReport {
            rolling_score,
            baseline_score: self.baseline_score,
            degradation,
            drifted: degradation > self.degradation_threshold,
        })
    }
}

/// Score a set of (prediction, actual) pairs with the given scorer.
fn score_pairs<S: Scorer>(scorer: &S, pairs: &[(f64, f64)]) -> Result<f64> {
    if pairs.is_empty() {
        return Err(DriftError::EmptyInput(
            "label-drift scoring needs at least one (prediction, actual) pair".into(),
        ));
    }
    let y_pred: Array1<f64> = pairs.iter().map(|&(p, _)| p).collect();
    let y_true: Array1<f64> = pairs.iter().map(|&(_, a)| a).collect();
    Ok(scorer.score(&y_true, &y_pred))
}
