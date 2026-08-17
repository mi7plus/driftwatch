//! Chi-square test of homogeneity for categorical drift.

use crate::binning::Histogram;
use crate::error::{DriftError, Result};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of a chi-square test of homogeneity between two categorical
/// distributions.
///
/// Carries the raw statistic and a p-value so callers apply their own
/// significance threshold. Complements PSI's magnitude-only signal with a
/// formal significance judgment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquareResult {
    /// The chi-square statistic `Σ (O − E)² / E` over the 2×k contingency table.
    pub statistic: f64,
    /// P-value for the null hypothesis that the reference and live category
    /// distributions are the same. Small p-values reject the null (drift).
    pub p_value: f64,
    /// Degrees of freedom used (`k − 1` over the categories present in at least
    /// one of the two samples).
    pub degrees_of_freedom: usize,
}

/// Chi-square test of homogeneity between a `reference` and `live` categorical
/// histogram.
///
/// The two histograms are treated as the rows of a 2×k contingency table (`k` =
/// number of categories) and tested for whether they could plausibly be samples
/// of the same underlying distribution. Categories absent from *both* samples
/// are dropped (they carry no information and would otherwise inflate the
/// degrees of freedom).
///
/// Because this uses observed *counts*, the histograms must carry counts, not
/// pre-normalized frequencies — which is exactly what [`Histogram`] stores.
///
/// # Errors
/// * [`DriftError::BinCountMismatch`] if the histograms have different bin
///   counts (their categories are not aligned — build both from a shared
///   [`ReferenceDistribution`](crate::ReferenceDistribution) to guarantee this).
/// * [`DriftError::EmptyInput`] if either sample has zero total count.
/// * [`DriftError::SampleTooSmall`] if fewer than two categories are present, so
///   there is no test to run (`df` would be zero).
pub fn chi_square_test(reference: &Histogram, live: &Histogram) -> Result<ChiSquareResult> {
    if reference.len() != live.len() {
        return Err(DriftError::BinCountMismatch {
            reference: reference.len(),
            live: live.len(),
        });
    }
    let ref_counts = reference.counts();
    let live_counts = live.counts();
    let ref_total = reference.total();
    let live_total = live.total();
    if ref_total <= 0.0 || live_total <= 0.0 {
        return Err(DriftError::EmptyInput(
            "chi-square test needs a non-empty reference and live sample".into(),
        ));
    }
    let grand = ref_total + live_total;

    let mut statistic = 0.0;
    let mut effective_categories = 0usize;
    for (&o_ref, &o_live) in ref_counts.iter().zip(live_counts) {
        let col_total = o_ref + o_live;
        if col_total <= 0.0 {
            // Category absent from both samples: no information, drop it.
            continue;
        }
        effective_categories += 1;
        let e_ref = ref_total * col_total / grand;
        let e_live = live_total * col_total / grand;
        statistic += (o_ref - e_ref).powi(2) / e_ref;
        statistic += (o_live - e_live).powi(2) / e_live;
    }

    if effective_categories < 2 {
        return Err(DriftError::SampleTooSmall {
            kind: "chi-square test",
            minimum: 2,
            actual: effective_categories,
        });
    }
    let df = effective_categories - 1;

    // statrs' ChiSquared is parameterized by freedom > 0, which holds here.
    let dist = ChiSquared::new(df as f64)
        .map_err(|e| DriftError::InvalidConfig(format!("chi-square distribution: {e}")))?;
    let p_value = 1.0 - dist.cdf(statistic);

    Ok(ChiSquareResult {
        statistic,
        p_value,
        degrees_of_freedom: df,
    })
}
