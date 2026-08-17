//! Binning: turning raw feature samples into comparable discretized
//! distributions.
//!
//! Every downstream binned drift metric (PSI, KL, JS, chi-square) operates on a
//! [`Histogram`]: a set of bins plus a per-bin count. The crucial correctness
//! rule is that live data must always be binned against the *reference's* chosen
//! bin edges (or category set), never rebinned independently — otherwise the
//! per-bin comparison is meaningless. [`ReferenceDistribution`] enforces that;
//! the strategies here only ever *fit* their edges on reference data.
//!
//! Two continuous strategies are provided:
//!
//! * [`EqualWidthBinning`] — fixed-width bins across the reference range.
//! * [`EqualFrequencyBinning`] — quantile bins so the reference data is spread
//!   evenly across bins. This is the more common convention for PSI in industry
//!   practice (credit-risk modeling, where PSI originates), and is what you
//!   should reach for by default.
//!
//! [`ReferenceDistribution`]: crate::ReferenceDistribution

mod equal_frequency;
mod equal_width;

pub use equal_frequency::EqualFrequencyBinning;
pub use equal_width::EqualWidthBinning;

use crate::error::{DriftError, Result};

/// Default number of bins for continuous strategies, matching the common PSI
/// convention of 10 bins.
pub const DEFAULT_BIN_COUNT: usize = 10;

/// How a [`Histogram`]'s bins are defined: continuous edges or discrete
/// categories.
#[derive(Clone, Debug, PartialEq)]
pub enum BinDefinition {
    /// Continuous bins described by `n + 1` sorted edges for `n` bins. The
    /// first and last edges are the observed reference min/max; values beyond
    /// them at transform time are clamped into the outermost bins.
    Continuous {
        /// Sorted bin edges, length `n_bins + 1`.
        edges: Vec<f64>,
    },
    /// Discrete category labels, one per bin, in a fixed order.
    Categorical {
        /// Category labels, one per bin.
        categories: Vec<String>,
    },
}

impl BinDefinition {
    /// Number of bins this definition describes.
    pub fn len(&self) -> usize {
        match self {
            BinDefinition::Continuous { edges } => edges.len().saturating_sub(1),
            BinDefinition::Categorical { categories } => categories.len(),
        }
    }

    /// Whether this definition describes zero bins.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A discretized distribution: a [`BinDefinition`] plus a raw count per bin.
///
/// Counts are kept as `f64` (rather than normalized away) so that a
/// significance test like chi-square, which needs observed counts, and a
/// magnitude metric like PSI, which needs proportions, can both be served from
/// the same type. Use [`Histogram::frequencies`] for the normalized view.
#[derive(Clone, Debug)]
pub struct Histogram {
    bins: BinDefinition,
    counts: Vec<f64>,
    total: f64,
}

impl Histogram {
    /// Build a histogram from a bin definition and matching per-bin counts.
    ///
    /// # Errors
    /// Returns [`DriftError::BinCountMismatch`] if `counts.len()` does not equal
    /// the number of bins in `bins`.
    pub fn new(bins: BinDefinition, counts: Vec<f64>) -> Result<Self> {
        if counts.len() != bins.len() {
            return Err(DriftError::BinCountMismatch {
                reference: bins.len(),
                live: counts.len(),
            });
        }
        let total = counts.iter().sum();
        Ok(Self {
            bins,
            counts,
            total,
        })
    }

    /// The bin definition backing this histogram.
    pub fn bins(&self) -> &BinDefinition {
        &self.bins
    }

    /// Raw per-bin counts.
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// Sum of all bin counts (the number of samples that went into the
    /// histogram).
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Number of bins.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether the histogram has zero bins.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Per-bin frequencies, normalized to sum to 1.0.
    ///
    /// If the histogram is empty (no samples), every frequency is `0.0` rather
    /// than `NaN`.
    pub fn frequencies(&self) -> Vec<f64> {
        if self.total <= 0.0 {
            return vec![0.0; self.counts.len()];
        }
        self.counts.iter().map(|&c| c / self.total).collect()
    }
}

/// Reject any non-finite value in a raw continuous sample slice.
pub(crate) fn ensure_finite(data: &[f64]) -> Result<()> {
    for (index, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(DriftError::NonFinite { index });
        }
    }
    Ok(())
}

/// Assign each sample to a bin defined by `edges` (length `n_bins + 1`),
/// clamping values below `edges[0]` into the first bin and values at or above
/// `edges[n]` into the last bin. Returns a continuous [`Histogram`].
///
/// This is the single place live data gets counted against a fixed set of
/// reference edges, guaranteeing reference and live histograms are always
/// comparable bin-for-bin.
pub(crate) fn histogram_from_edges(edges: &[f64], data: &[f64]) -> Result<Histogram> {
    ensure_finite(data)?;
    let n_bins = edges.len().saturating_sub(1);
    if n_bins == 0 {
        return Err(DriftError::InvalidBinCount(n_bins));
    }
    let mut counts = vec![0.0f64; n_bins];
    for &v in data {
        // Clamp out-of-range values into the outermost bins: novel extremes in
        // live data are a drift signal, not an error.
        let idx = if v <= edges[0] {
            0
        } else if v >= edges[n_bins] {
            n_bins - 1
        } else {
            // edges are sorted; find the bin whose [lo, hi) contains v.
            match edges.partition_point(|&e| e <= v) {
                0 => 0,
                p if p >= n_bins => n_bins - 1,
                p => p - 1,
            }
        };
        counts[idx] += 1.0;
    }
    Histogram::new(
        BinDefinition::Continuous {
            edges: edges.to_vec(),
        },
        counts,
    )
}

/// Strategy for choosing continuous bin edges from reference data.
///
/// Implementors only ever see reference data. The resulting edges are then
/// reused verbatim for every live window.
pub trait ContinuousBinning {
    /// Compute sorted bin edges (length `n_bins + 1`) from reference samples.
    ///
    /// # Errors
    /// Returns [`DriftError::EmptyInput`] on empty data,
    /// [`DriftError::NonFinite`] on a non-finite value, and
    /// [`DriftError::ZeroWidthRange`] when every reference value is identical
    /// (no non-degenerate edges exist).
    fn fit_edges(&self, reference: &[f64]) -> Result<Vec<f64>>;
}
