//! Kullback–Leibler divergence.

use super::{smoothed_pair, DEFAULT_EPSILON};
use crate::binning::Histogram;
use crate::error::Result;

/// Kullback–Leibler divergence of the `live` distribution *from* the `reference`
/// distribution, in nats.
///
/// ```text
/// D_KL(live ‖ reference) = Σ_bin live_i · ln(live_i / ref_i)
/// ```
///
/// # Directional asymmetry
///
/// KL is **not symmetric**: `D_KL(live ‖ reference) ≠ D_KL(reference ‖ live)`.
/// This function fixes the direction as `D_KL(live ‖ reference)` — it weights
/// each bin by the *live* probability, answering "how surprising is the live
/// data under the reference model?", which is the natural question when the
/// reference is the trusted baseline. If you need the other direction, swap the
/// arguments. For a symmetric, bounded alternative use
/// [`js_divergence`](crate::js_divergence).
///
/// Both distributions are epsilon-smoothed first (see
/// [`DEFAULT_EPSILON`](crate::metrics::DEFAULT_EPSILON)), so an empty bin never
/// yields `inf`/`NaN`.
///
/// # Errors
/// Returns [`DriftError::BinCountMismatch`](crate::DriftError::BinCountMismatch)
/// if the histograms have different bin counts.
pub fn kl_divergence(reference: &Histogram, live: &Histogram) -> Result<f64> {
    kl_divergence_with_epsilon(reference, live, DEFAULT_EPSILON)
}

/// [`kl_divergence`] with an explicit epsilon smoothing constant.
///
/// # Errors
/// Same as [`kl_divergence`], plus
/// [`DriftError::InvalidConfig`](crate::DriftError::InvalidConfig) if `epsilon`
/// is negative or non-finite.
pub fn kl_divergence_with_epsilon(
    reference: &Histogram,
    live: &Histogram,
    epsilon: f64,
) -> Result<f64> {
    let (r, l) = smoothed_pair(reference, live, epsilon)?;
    Ok(kl(&l, &r))
}

/// Raw KL divergence `D_KL(p ‖ q)` over two smoothed probability vectors.
pub(crate) fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| pi * (pi / qi).ln()).sum()
}
