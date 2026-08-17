//! Equal-width continuous binning.

use super::{ContinuousBinning, DEFAULT_BIN_COUNT};
use crate::error::{DriftError, Result};

/// Fixed-width bins spanning the reference data's observed `[min, max]` range.
///
/// Simple and fast, but sensitive to outliers (a single extreme value stretches
/// every bin). For PSI specifically, prefer [`EqualFrequencyBinning`], which is
/// the industry convention; equal-width is offered mainly for KL/JS and cases
/// where fixed, human-readable bin boundaries are wanted.
///
/// [`EqualFrequencyBinning`]: super::EqualFrequencyBinning
#[derive(Clone, Copy, Debug)]
pub struct EqualWidthBinning {
    n_bins: usize,
}

impl EqualWidthBinning {
    /// Equal-width binning with an explicit bin count.
    ///
    /// # Errors
    /// Returns [`DriftError::InvalidBinCount`] if `n_bins` is zero.
    pub fn new(n_bins: usize) -> Result<Self> {
        if n_bins == 0 {
            return Err(DriftError::InvalidBinCount(n_bins));
        }
        Ok(Self { n_bins })
    }

    /// Number of bins this strategy will produce.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}

impl Default for EqualWidthBinning {
    /// [`DEFAULT_BIN_COUNT`] (10) bins, the common PSI convention.
    fn default() -> Self {
        Self {
            n_bins: DEFAULT_BIN_COUNT,
        }
    }
}

impl ContinuousBinning for EqualWidthBinning {
    fn fit_edges(&self, reference: &[f64]) -> Result<Vec<f64>> {
        if reference.is_empty() {
            return Err(DriftError::EmptyInput(
                "equal-width binning needs at least one reference value".into(),
            ));
        }
        super::ensure_finite(reference)?;

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in reference {
            min = min.min(v);
            max = max.max(v);
        }
        if min == max {
            return Err(DriftError::ZeroWidthRange { value: min });
        }

        let width = (max - min) / self.n_bins as f64;
        let mut edges = Vec::with_capacity(self.n_bins + 1);
        for i in 0..self.n_bins {
            edges.push(min + width * i as f64);
        }
        // Set the final edge exactly to `max` to avoid floating-point drift.
        edges.push(max);
        Ok(edges)
    }
}
