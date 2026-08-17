//! Edge cases across the crate, each with a documented expected behavior.

use driftwatch::{
    DatasetMonitor, DriftError, EqualFrequencyBinning, LiveFeature, LiveWindow,
    ReferenceDistribution, WindowMode,
};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn empty_live_window_snapshot_is_empty() {
    let window = LiveWindow::new(WindowMode::Count(10));
    assert!(window.snapshot().is_empty());
    assert!(window.is_empty());
}

#[test]
fn empty_live_batch_is_rejected() {
    let reference = ReferenceDistribution::fit_continuous(
        "x",
        &linspace(0.0, 1.0, 100),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap();
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(reference);

    let empty: [f64; 0] = [];
    let err = monitor.check(&[("x", LiveFeature::Continuous(&empty))]);
    assert!(matches!(err, Err(DriftError::SampleTooSmall { .. })));
}

#[test]
fn below_minimum_sample_size_errors() {
    let reference = ReferenceDistribution::fit_continuous(
        "x",
        &linspace(0.0, 1.0, 100),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap();
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(reference);
    let monitor = monitor.with_min_live_samples(50);

    let too_few = linspace(0.0, 1.0, 5);
    let err = monitor.check(&[("x", LiveFeature::Continuous(&too_few))]);
    assert!(matches!(
        err,
        Err(DriftError::SampleTooSmall {
            minimum: 50,
            actual: 5,
            ..
        })
    ));
}

#[test]
fn single_category_reference_is_usable() {
    // Reference has exactly one category. PSI over a single bin is 0; the
    // chi-square test doesn't apply (needs >= 2 categories) and is silently
    // dropped rather than failing the check.
    let reference = ReferenceDistribution::fit_categorical("k", &["a", "a", "a"]).unwrap();
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(reference);

    let live = ["a", "a"];
    let report = monitor
        .check(&[("k", LiveFeature::Categorical(&live))])
        .unwrap();
    assert!(!report.features[0].drifted());
    assert!(report.features[0]
        .score(driftwatch::MetricKind::ChiSquare)
        .is_none());
}

#[test]
fn all_identical_continuous_values_cannot_form_bins() {
    // Zero-width range: no non-degenerate set of bin edges exists, so fitting a
    // continuous reference on all-identical values is a documented error.
    let flat = vec![7.0_f64; 100];
    let result =
        ReferenceDistribution::fit_continuous("x", &flat, EqualFrequencyBinning::new(10).unwrap());
    assert!(matches!(result, Err(DriftError::ZeroWidthRange { .. })));
}
