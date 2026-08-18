//! Continuously-updated online drift (feature `streaming`).
//!
//! The batch [`DatasetMonitor`](crate::DatasetMonitor) recomputes drift over
//! discrete windows/snapshots. This module is the *online* counterpart: an
//! [`OnlineDistribution`] absorbs values one at a time into a quantile sketch
//! (DDSketch) in bounded memory, so drift against the reference is queryable at
//! any instant without buffering raw samples. A [`StreamingMonitor`] tracks one
//! online distribution per feature; a [`PageHinkleyDetector`] complements it
//! with classic online change-point detection on a scalar signal.
//!
//! Because the live distribution is reconstructed from a sketch, online PSI is
//! **approximate** — it converges to the batch value within the sketch's
//! accuracy, and improves with more probe points. Exact two-sample tests like
//! KS remain a windowed operation and are not offered here.

mod page_hinkley;

pub use page_hinkley::{PageHinkleyChange, PageHinkleyDetector};

use crate::binning::{BinDefinition, Histogram};
use crate::distribution::ReferenceDistribution;
use crate::error::{DriftError, Result};
use crate::metrics::psi;
use sketches_ddsketch::{Config, DDSketch};

/// Default number of quantile probes used to reconstruct a histogram from the
/// sketch. More probes give a smoother, more accurate reconstruction.
pub const DEFAULT_PROBES: usize = 512;

/// An online estimate of a continuous feature's distribution, backed by a
/// DDSketch. Values are absorbed one at a time in bounded memory.
pub struct OnlineDistribution {
    sketch: DDSketch,
    probes: usize,
}

impl Default for OnlineDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineDistribution {
    /// A new, empty online distribution with default accuracy and
    /// [`DEFAULT_PROBES`] probe points.
    pub fn new() -> Self {
        Self {
            sketch: DDSketch::new(Config::defaults()),
            probes: DEFAULT_PROBES,
        }
    }

    /// Set the number of quantile probes used when reconstructing a histogram.
    pub fn with_probes(mut self, probes: usize) -> Self {
        self.probes = probes.max(1);
        self
    }

    /// Absorb one value. Non-finite values are ignored (they cannot be sketched).
    pub fn update(&mut self, value: f64) {
        if value.is_finite() {
            self.sketch.add(value);
        }
    }

    /// Number of values absorbed so far.
    pub fn count(&self) -> usize {
        self.sketch.count()
    }

    /// Whether no values have been absorbed.
    pub fn is_empty(&self) -> bool {
        self.sketch.count() == 0
    }

    /// Reconstruct an approximate histogram over the given reference `edges` by
    /// probing the sketch at evenly-spaced quantiles.
    ///
    /// Each of `probes` probes contributes equal mass to the bin its value falls
    /// into (values outside the edges clamp into the outermost bins), so the
    /// result is a valid, comparable [`Histogram`] against a reference fitted on
    /// the same edges.
    pub fn snapshot_histogram(&self, edges: &[f64]) -> Result<Histogram> {
        let n_bins = edges.len().saturating_sub(1);
        if n_bins == 0 {
            return Err(DriftError::InvalidBinCount(n_bins));
        }
        let mut counts = vec![0.0f64; n_bins];
        if self.sketch.count() > 0 {
            for k in 0..self.probes {
                let q = (k as f64 + 0.5) / self.probes as f64;
                if let Ok(Some(v)) = self.sketch.quantile(q) {
                    counts[bin_index(edges, v, n_bins)] += 1.0;
                }
            }
        }
        Histogram::new(
            BinDefinition::Continuous {
                edges: edges.to_vec(),
            },
            counts,
        )
    }
}

/// Clamp a value into a bin index over sorted `edges` (length `n_bins + 1`).
fn bin_index(edges: &[f64], v: f64, n_bins: usize) -> usize {
    if v <= edges[0] {
        0
    } else if v >= edges[n_bins] {
        n_bins - 1
    } else {
        match edges.partition_point(|&e| e <= v) {
            0 => 0,
            p if p >= n_bins => n_bins - 1,
            p => p - 1,
        }
    }
}

struct StreamFeature {
    name: String,
    reference_hist: Histogram,
    edges: Vec<f64>,
    online: OnlineDistribution,
    threshold: f64,
}

/// Tracks one [`OnlineDistribution`] per continuous feature and reports online
/// PSI against each feature's reference at any moment.
pub struct StreamingMonitor {
    features: Vec<StreamFeature>,
    dataset_fraction_threshold: f64,
}

impl Default for StreamingMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMonitor {
    /// A new streaming monitor with the default dataset-drift fraction threshold.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            dataset_fraction_threshold: 0.5,
        }
    }

    /// Set the fraction-of-features threshold for the aggregate verdict.
    pub fn with_dataset_fraction_threshold(mut self, fraction: f64) -> Self {
        self.dataset_fraction_threshold = fraction;
        self
    }

    /// Register a continuous feature from its reference distribution, with a PSI
    /// drift threshold.
    ///
    /// # Errors
    /// [`DriftError::InvalidConfig`] if `reference` is categorical (online drift
    /// here is defined for continuous features).
    pub fn add_feature(&mut self, reference: &ReferenceDistribution, threshold: f64) -> Result<()> {
        let edges = match reference.histogram().bins() {
            BinDefinition::Continuous { edges } => edges.clone(),
            BinDefinition::Categorical { .. } => {
                return Err(DriftError::InvalidConfig(format!(
                    "streaming feature '{}' must be continuous",
                    reference.name()
                )))
            }
        };
        self.features.push(StreamFeature {
            name: reference.name().to_string(),
            reference_hist: reference.histogram().clone(),
            edges,
            online: OnlineDistribution::new(),
            threshold,
        });
        Ok(())
    }

    /// Absorb one live value for a named feature.
    ///
    /// # Errors
    /// [`DriftError::UnknownFeature`] if no feature by that name is registered.
    pub fn update(&mut self, feature: &str, value: f64) -> Result<()> {
        let f = self
            .features
            .iter_mut()
            .find(|f| f.name == feature)
            .ok_or_else(|| DriftError::UnknownFeature(feature.to_string()))?;
        f.online.update(value);
        Ok(())
    }

    /// Current online PSI for one feature.
    ///
    /// # Errors
    /// [`DriftError::UnknownFeature`] if the feature is not registered, plus any
    /// metric error.
    pub fn feature_psi(&self, feature: &str) -> Result<f64> {
        let f = self
            .features
            .iter()
            .find(|f| f.name == feature)
            .ok_or_else(|| DriftError::UnknownFeature(feature.to_string()))?;
        let live = f.online.snapshot_histogram(&f.edges)?;
        psi(&f.reference_hist, &live)
    }

    /// Compute a snapshot report of online drift across every feature.
    ///
    /// # Errors
    /// Propagates any metric error.
    pub fn report(&self) -> Result<StreamingReport> {
        let mut features = Vec::with_capacity(self.features.len());
        for f in &self.features {
            let live = f.online.snapshot_histogram(&f.edges)?;
            let psi_value = psi(&f.reference_hist, &live)?;
            features.push(StreamFeatureDrift {
                feature: f.name.clone(),
                psi: psi_value,
                count: f.online.count(),
                threshold: f.threshold,
                drifted: psi_value > f.threshold,
            });
        }
        Ok(StreamingReport {
            features,
            dataset_fraction_threshold: self.dataset_fraction_threshold,
        })
    }
}

/// A single feature's online drift result.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamFeatureDrift {
    /// Feature name.
    pub feature: String,
    /// Current online PSI against the reference.
    pub psi: f64,
    /// Number of values absorbed so far.
    pub count: usize,
    /// The PSI threshold applied.
    pub threshold: f64,
    /// Whether the feature is currently drifting.
    pub drifted: bool,
}

/// A snapshot of online drift across all monitored features.
#[derive(Clone, Debug)]
pub struct StreamingReport {
    /// Per-feature online drift.
    pub features: Vec<StreamFeatureDrift>,
    /// Fraction-of-features threshold for the aggregate verdict.
    pub dataset_fraction_threshold: f64,
}

impl StreamingReport {
    /// Fraction of features currently drifting, in `[0, 1]`.
    pub fn drifted_fraction(&self) -> f64 {
        if self.features.is_empty() {
            return 0.0;
        }
        self.features.iter().filter(|f| f.drifted).count() as f64 / self.features.len() as f64
    }

    /// Whether the aggregate dataset-drift verdict is positive.
    pub fn dataset_drift_detected(&self) -> bool {
        self.drifted_fraction() > self.dataset_fraction_threshold
    }
}
