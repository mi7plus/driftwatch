//! Two-sample Kolmogorov–Smirnov test.

use crate::binning::ensure_finite;
use crate::error::{DriftError, Result};

/// Result of a two-sample Kolmogorov–Smirnov test.
///
/// Carries both the raw statistic and a p-value so callers apply their own
/// significance threshold rather than the crate baking one in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KsTestResult {
    /// The KS statistic `D`: the maximum absolute difference between the two
    /// empirical CDFs. In `[0, 1]`; larger means more divergence.
    pub statistic: f64,
    /// Asymptotic two-sided p-value for the null hypothesis that both samples
    /// are drawn from the same distribution. Small p-values reject the null
    /// (i.e. indicate drift).
    pub p_value: f64,
    /// Number of reference samples.
    pub n_reference: usize,
    /// Number of live samples.
    pub n_live: usize,
}

/// Two-sample Kolmogorov–Smirnov test comparing raw `reference` and `live`
/// continuous samples.
///
/// Unlike the binned metrics ([`psi`](crate::psi) et al.), this operates
/// directly on raw samples, so its result does **not** depend on any
/// binning-strategy choice — a useful property when you want a drift signal
/// that is robust to how you would have discretized the feature.
///
/// The p-value uses the standard asymptotic Kolmogorov distribution
/// approximation (Numerical Recipes' `probks`); it is accurate for
/// moderate-to-large samples and conservative for very small ones.
///
/// # Errors
/// Returns [`DriftError::EmptyInput`] if either slice is empty, or
/// [`DriftError::NonFinite`] on a `NaN`/infinite value.
///
/// # Example
/// ```
/// use driftwatch::ks_test;
/// let reference = [0.1, 0.2, 0.3, 0.4, 0.5];
/// let live = [0.1, 0.2, 0.3, 0.4, 0.5];
/// let r = ks_test(&reference, &live).unwrap();
/// assert_eq!(r.statistic, 0.0); // identical samples
/// ```
pub fn ks_test(reference: &[f64], live: &[f64]) -> Result<KsTestResult> {
    if reference.is_empty() || live.is_empty() {
        return Err(DriftError::EmptyInput(
            "KS test needs at least one sample in each group".into(),
        ));
    }
    ensure_finite(reference)?;
    ensure_finite(live)?;

    let mut a = reference.to_vec();
    let mut b = live.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).expect("finiteness checked"));
    b.sort_by(|x, y| x.partial_cmp(y).expect("finiteness checked"));

    let (n, m) = (a.len(), b.len());
    let (en1, en2) = (n as f64, m as f64);
    let mut i = 0usize;
    let mut j = 0usize;
    let mut fn1 = 0.0;
    let mut fn2 = 0.0;
    let mut d = 0.0f64;

    while i < n && j < m {
        let d1 = a[i];
        let d2 = b[j];
        if d1 <= d2 {
            while i < n && a[i] == d1 {
                i += 1;
            }
            fn1 = i as f64 / en1;
        }
        if d2 <= d1 {
            while j < m && b[j] == d2 {
                j += 1;
            }
            fn2 = j as f64 / en2;
        }
        let dt = (fn2 - fn1).abs();
        if dt > d {
            d = dt;
        }
    }

    let en = (en1 * en2 / (en1 + en2)).sqrt();
    let p_value = ks_prob((en + 0.12 + 0.11 / en) * d);

    Ok(KsTestResult {
        statistic: d,
        p_value,
        n_reference: n,
        n_live: m,
    })
}

/// Kolmogorov distribution survival function `Q_KS(λ)`, via the alternating
/// series `2 Σ_{k≥1} (−1)^{k−1} e^{−2 k² λ²}`. Returns a probability in `[0, 1]`.
fn ks_prob(lambda: f64) -> f64 {
    const EPS1: f64 = 1e-6;
    const EPS2: f64 = 1e-10;
    if lambda <= 0.0 {
        return 1.0;
    }
    let a2 = -2.0 * lambda * lambda;
    let mut fac = 2.0;
    let mut sum = 0.0;
    let mut termbf = 0.0;
    for k in 1..=100 {
        let term = fac * (a2 * (k * k) as f64).exp();
        sum += term;
        if term.abs() <= EPS1 * termbf || term.abs() <= EPS2 * sum {
            return sum.clamp(0.0, 1.0);
        }
        fac = -fac;
        termbf = term.abs();
    }
    // Series failed to converge (λ extremely small): p ≈ 1.
    1.0
}
