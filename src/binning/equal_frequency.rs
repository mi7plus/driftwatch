//! Equal-frequency (quantile) continuous binning — the default PSI convention.

use super::{ContinuousBinning, DEFAULT_BIN_COUNT};
use crate::error::{DriftError, Result};

/// Quantile bins chosen so the *reference* data is spread evenly across bins.
///
/// This is the more common convention for PSI in industry practice (it
/// originates in credit-risk modeling), because it puts equal reference mass in
/// every bin and so gives each bin comparable statistical weight regardless of
/// how skewed the feature is. Prefer this over [`EqualWidthBinning`] for PSI.
///
/// Bin edges are computed as empirical quantiles of the reference sample with
/// linear interpolation. If the reference data has heavy ties (many identical
/// values), adjacent quantile edges can coincide; such duplicate edges are
/// collapsed, which reduces the effective bin count. That collapse is applied
/// consistently to every live window, so reference and live histograms stay
/// comparable.
///
/// [`EqualWidthBinning`]: super::EqualWidthBinning
#[derive(Clone, Copy, Debug)]
pub struct EqualFrequencyBinning {
    n_bins: usize,
}

impl EqualFrequencyBinning {
    /// Equal-frequency binning with an explicit bin count.
    ///
    /// # Errors
    /// Returns [`DriftError::InvalidBinCount`] if `n_bins` is zero.
    pub fn new(n_bins: usize) -> Result<Self> {
        if n_bins == 0 {
            return Err(DriftError::InvalidBinCount(n_bins));
        }
        Ok(Self { n_bins })
    }

    /// Number of bins this strategy will produce (before any tie collapse).
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}

impl Default for EqualFrequencyBinning {
    /// [`DEFAULT_BIN_COUNT`] (10) bins, the common PSI convention.
    fn default() -> Self {
        Self {
            n_bins: DEFAULT_BIN_COUNT,
        }
    }
}

/// Empirical quantile of an already-sorted slice, using the common
/// linear-interpolation ("type 7") definition.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl ContinuousBinning for EqualFrequencyBinning {
    fn fit_edges(&self, reference: &[f64]) -> Result<Vec<f64>> {
        if reference.is_empty() {
            return Err(DriftError::EmptyInput(
                "equal-frequency binning needs at least one reference value".into(),
            ));
        }
        super::ensure_finite(reference)?;

        let mut sorted = reference.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finiteness checked above"));

        if sorted[0] == sorted[sorted.len() - 1] {
            return Err(DriftError::ZeroWidthRange { value: sorted[0] });
        }

        let mut edges = Vec::with_capacity(self.n_bins + 1);
        edges.push(sorted[0]);
        for i in 1..self.n_bins {
            let q = i as f64 / self.n_bins as f64;
            edges.push(quantile_sorted(&sorted, q));
        }
        edges.push(sorted[sorted.len() - 1]);

        // Collapse duplicate edges produced by ties, keeping strict monotonicity.
        edges.dedup();
        if edges.len() < 2 {
            return Err(DriftError::ZeroWidthRange { value: sorted[0] });
        }
        Ok(edges)
    }
}
