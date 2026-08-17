//! `DatasetMonitor` must flag only the features that actually drifted and roll
//! them up into an accurate aggregate verdict.

use driftwatch::{DatasetMonitor, EqualFrequencyBinning, LiveFeature, ReferenceDistribution};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

fn continuous_ref(name: &str, lo: f64, hi: f64) -> ReferenceDistribution {
    ReferenceDistribution::fit_continuous(
        name,
        &linspace(lo, hi, 500),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap()
}

#[test]
fn flags_only_shifted_features() {
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(continuous_ref("a", 0.0, 1.0));
    monitor.add_feature(continuous_ref("b", 0.0, 1.0));
    monitor.add_feature(continuous_ref("c", 0.0, 1.0));
    monitor.add_feature(continuous_ref("d", 0.0, 1.0));

    // a, b unchanged; c, d shifted hard.
    let unchanged = linspace(0.0, 1.0, 500);
    let shifted = linspace(1.0, 2.0, 500);

    let report = monitor
        .check(&[
            ("a", LiveFeature::Continuous(&unchanged)),
            ("b", LiveFeature::Continuous(&unchanged)),
            ("c", LiveFeature::Continuous(&shifted)),
            ("d", LiveFeature::Continuous(&shifted)),
        ])
        .unwrap();

    let drifted: Vec<&str> = report
        .drifted_features()
        .map(|f| f.feature.as_str())
        .collect();
    assert_eq!(drifted, vec!["c", "d"]);

    // 2 of 4 drifted = 0.5, which is not strictly greater than the 0.5 default,
    // so the dataset verdict is stable.
    assert!((report.drifted_fraction() - 0.5).abs() < 1e-9);
    assert!(!report.dataset_drift_detected());
}

#[test]
fn aggregate_dataset_drift_when_majority_shift() {
    let mut monitor = DatasetMonitor::new();
    for name in ["a", "b", "c", "d"] {
        monitor.add_feature(continuous_ref(name, 0.0, 1.0));
    }
    let unchanged = linspace(0.0, 1.0, 500);
    let shifted = linspace(1.0, 2.0, 500);

    // 3 of 4 drift → fraction 0.75 > 0.5 → dataset drift.
    let report = monitor
        .check(&[
            ("a", LiveFeature::Continuous(&unchanged)),
            ("b", LiveFeature::Continuous(&shifted)),
            ("c", LiveFeature::Continuous(&shifted)),
            ("d", LiveFeature::Continuous(&shifted)),
        ])
        .unwrap();
    assert!(report.dataset_drift_detected());
    assert_eq!(report.drifted_features().count(), 3);
}

#[test]
fn missing_feature_is_an_error() {
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(continuous_ref("a", 0.0, 1.0));
    let err = monitor.check(&[]);
    assert!(err.is_err());
}

#[test]
fn categorical_feature_monitored_end_to_end() {
    let reference =
        ReferenceDistribution::fit_categorical("color", &["red", "red", "blue", "green"]).unwrap();
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(reference);

    // Live distribution dominated by a novel category → should drift.
    let live = ["yellow", "yellow", "yellow", "red"];
    let report = monitor
        .check(&[("color", LiveFeature::Categorical(&live))])
        .unwrap();
    assert!(report.features[0].primary_statistic().is_finite());
    // A chi-square score is reported alongside PSI.
    assert!(report.features[0]
        .score(driftwatch::MetricKind::ChiSquare)
        .is_some());
}

#[test]
fn report_display_is_readable() {
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(continuous_ref("score", 0.0, 1.0));
    let shifted = linspace(1.0, 2.0, 500);
    let report = monitor
        .check(&[("score", LiveFeature::Continuous(&shifted))])
        .unwrap();
    let text = format!("{report}");
    assert!(text.contains("score"));
    assert!(text.contains("dataset"));
}
