//! Drift metrics.
//!
//! Two families live here:
//!
//! * **Binned metrics** over [`Histogram`]s —
//!   [`psi`], [`kl_divergence`], [`js_divergence`]. All three are undefined when
//!   a bin is empty in one distribution, so every one applies epsilon smoothing
//!   (see [`DEFAULT_EPSILON`]) before computing. Because they consume
//!   histograms, their results depend on the binning strategy chosen upstream.
//!
//! * **Raw-sample tests** — [`ks_test`] (continuous) and [`chi_square_test`]
//!   (categorical). These return a statistic *and* a p-value so you can apply
//!   your own significance threshold. The KS test operates on raw samples, so —
//!   unlike the binned metrics — its result is not sensitive to any
//!   binning-strategy choice, which is a genuine advantage when you want a
//!   drift signal independent of that decision.

mod chi_square_test;
mod js_divergence;
mod kl_divergence;
mod ks_test;
mod psi;

pub use chi_square_test::{chi_square_test, ChiSquareResult};
pub use js_divergence::js_divergence;
pub use kl_divergence::kl_divergence;
pub use ks_test::{ks_test, KsTestResult};
pub use psi::psi;

pub use psi::psi_with_epsilon;
pub use {js_divergence::js_divergence_with_epsilon, kl_divergence::kl_divergence_with_epsilon};

use crate::binning::Histogram;
use crate::error::{DriftError, Result};

/// Default epsilon smoothing constant applied to every bin before computing PSI,
/// KL, or JS divergence.
///
/// Without smoothing, an empty bin in either distribution makes these metrics
/// `inf`/`NaN` (they take `ln(0)` or divide by zero). Smoothing adds this small
/// constant to every bin's probability and renormalizes, which keeps the metric
/// finite while barely perturbing well-populated bins. It is configurable via
/// the `*_with_epsilon` variants.
pub const DEFAULT_EPSILON: f64 = 1e-6;

/// Return the two histograms' smoothed probability vectors, or an error if their
/// bin counts differ.
///
/// Smoothing is additive: `p_i = (f_i + epsilon) / (1 + n * epsilon)`, where
/// `f_i` is bin `i`'s raw frequency and `n` the bin count. The result is a
/// proper probability vector (sums to 1) with no zero entries.
pub(crate) fn smoothed_pair(
    reference: &Histogram,
    live: &Histogram,
    epsilon: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if reference.len() != live.len() {
        return Err(DriftError::BinCountMismatch {
            reference: reference.len(),
            live: live.len(),
        });
    }
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(DriftError::InvalidConfig(format!(
            "epsilon must be finite and non-negative, got {epsilon}"
        )));
    }
    Ok((
        smooth(&reference.frequencies(), epsilon),
        smooth(&live.frequencies(), epsilon),
    ))
}

fn smooth(freqs: &[f64], epsilon: f64) -> Vec<f64> {
    let n = freqs.len() as f64;
    let denom = 1.0 + epsilon * n;
    freqs.iter().map(|&f| (f + epsilon) / denom).collect()
}
