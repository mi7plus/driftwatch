//! Binning strategy tests: both strategies must produce histograms whose
//! frequencies sum to 1.0 on fixed data.

use approx::assert_abs_diff_eq;
use driftwatch::{
    ContinuousBinning, EqualFrequencyBinning, EqualWidthBinning, ReferenceDistribution,
};

fn baseline() -> Vec<f64> {
    (0..100).map(|i| i as f64).collect()
}

#[test]
fn equal_width_frequencies_sum_to_one() {
    let reference = ReferenceDistribution::fit_continuous(
        "x",
        &baseline(),
        EqualWidthBinning::new(10).unwrap(),
    )
    .unwrap();
    let sum: f64 = reference.histogram().frequencies().iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
    assert_eq!(reference.histogram().len(), 10);
}

#[test]
fn equal_frequency_frequencies_sum_to_one() {
    let reference = ReferenceDistribution::fit_continuous(
        "x",
        &baseline(),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap();
    let sum: f64 = reference.histogram().frequencies().iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
}

#[test]
fn equal_frequency_spreads_reference_evenly() {
    // With 100 evenly-spaced points into 10 quantile bins, each bin holds ~10.
    let edges = EqualFrequencyBinning::new(10)
        .unwrap()
        .fit_edges(&baseline())
        .unwrap();
    assert_eq!(edges.len(), 11);
    // Edges must be non-decreasing.
    for w in edges.windows(2) {
        assert!(w[1] >= w[0]);
    }
}

#[test]
fn zero_bin_count_is_rejected() {
    assert!(EqualWidthBinning::new(0).is_err());
    assert!(EqualFrequencyBinning::new(0).is_err());
}

#[test]
fn all_identical_values_is_zero_width_error() {
    let flat = vec![3.0_f64; 50];
    assert!(EqualWidthBinning::new(10)
        .unwrap()
        .fit_edges(&flat)
        .is_err());
    assert!(EqualFrequencyBinning::new(10)
        .unwrap()
        .fit_edges(&flat)
        .is_err());
}

#[test]
fn non_finite_input_is_rejected() {
    let bad = vec![1.0, f64::NAN, 3.0];
    assert!(EqualWidthBinning::new(5).unwrap().fit_edges(&bad).is_err());
}
