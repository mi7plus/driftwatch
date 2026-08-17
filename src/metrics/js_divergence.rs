//! Jensen–Shannon divergence.

use super::kl_divergence::kl;
use super::{smoothed_pair, DEFAULT_EPSILON};
use crate::binning::Histogram;
use crate::error::Result;

/// Jensen–Shannon divergence between a `reference` and `live` histogram, in nats.
///
/// ```text
/// M = ½ (live + reference)
/// JS = ½ D_KL(live ‖ M) + ½ D_KL(reference ‖ M)
/// ```
///
/// Unlike [`kl_divergence`](crate::kl_divergence), JS is **symmetric** and
/// **bounded** to `[0, ln 2]` (≈ `[0, 0.693]`) when computed in nats. Reach for
/// it when you want a single symmetric, bounded score that is directly
/// comparable across features — PSI and KL are unbounded and so harder to
/// compare on a common scale.
///
/// Both distributions are epsilon-smoothed first (see
/// [`DEFAULT_EPSILON`](crate::metrics::DEFAULT_EPSILON)).
///
/// # Errors
/// Returns [`DriftError::BinCountMismatch`](crate::DriftError::BinCountMismatch)
/// if the histograms have different bin counts.
pub fn js_divergence(reference: &Histogram, live: &Histogram) -> Result<f64> {
    js_divergence_with_epsilon(reference, live, DEFAULT_EPSILON)
}

/// [`js_divergence`] with an explicit epsilon smoothing constant.
///
/// # Errors
/// Same as [`js_divergence`], plus
/// [`DriftError::InvalidConfig`](crate::DriftError::InvalidConfig) if `epsilon`
/// is negative or non-finite.
pub fn js_divergence_with_epsilon(
    reference: &Histogram,
    live: &Histogram,
    epsilon: f64,
) -> Result<f64> {
    let (r, l) = smoothed_pair(reference, live, epsilon)?;
    let m: Vec<f64> = r.iter().zip(&l).map(|(&rp, &lp)| 0.5 * (rp + lp)).collect();
    Ok(0.5 * kl(&l, &m) + 0.5 * kl(&r, &m))
}
