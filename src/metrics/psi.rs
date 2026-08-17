//! Population Stability Index (PSI).

use super::{smoothed_pair, DEFAULT_EPSILON};
use crate::binning::Histogram;
use crate::error::Result;

/// Population Stability Index between a `reference` and `live` histogram.
///
/// PSI is the workhorse drift metric from credit-risk modeling:
///
/// ```text
/// PSI = Σ_bin (live_pct − ref_pct) · ln(live_pct / ref_pct)
/// ```
///
/// Both distributions are epsilon-smoothed (see
/// [`DEFAULT_EPSILON`](crate::metrics::DEFAULT_EPSILON)) first, so an empty bin
/// never yields `inf`/`NaN`.
///
/// # Threshold guidance
///
/// These commonly-cited bands are **convention, not mathematical fact** — treat
/// them as a starting point and calibrate to your own data:
///
/// | PSI          | Interpretation             |
/// |--------------|----------------------------|
/// | `< 0.10`     | no significant change      |
/// | `0.10–0.25`  | moderate change            |
/// | `> 0.25`     | significant change         |
///
/// # Errors
/// Returns [`DriftError::BinCountMismatch`](crate::DriftError::BinCountMismatch)
/// if the two histograms have different bin counts.
///
/// # Example
/// ```
/// use driftwatch::{psi, Histogram, BinDefinition};
///
/// let bins = BinDefinition::Continuous { edges: vec![0.0, 1.0, 2.0] };
/// let reference = Histogram::new(bins.clone(), vec![50.0, 50.0]).unwrap();
/// let live = Histogram::new(bins, vec![50.0, 50.0]).unwrap();
/// assert!(psi(&reference, &live).unwrap() < 1e-9); // identical → ~0
/// ```
pub fn psi(reference: &Histogram, live: &Histogram) -> Result<f64> {
    psi_with_epsilon(reference, live, DEFAULT_EPSILON)
}

/// [`psi`] with an explicit epsilon smoothing constant.
///
/// # Errors
/// Same as [`psi`], plus
/// [`DriftError::InvalidConfig`](crate::DriftError::InvalidConfig) if `epsilon`
/// is negative or non-finite.
pub fn psi_with_epsilon(reference: &Histogram, live: &Histogram, epsilon: f64) -> Result<f64> {
    let (r, l) = smoothed_pair(reference, live, epsilon)?;
    let psi = r
        .iter()
        .zip(&l)
        .map(|(&rp, &lp)| (lp - rp) * (lp / rp).ln())
        .sum();
    Ok(psi)
}
