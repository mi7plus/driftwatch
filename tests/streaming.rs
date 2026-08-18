//! Online/streaming drift: sketch-backed distributions and Page-Hinkley.
#![cfg(feature = "streaming")]

use driftwatch::streaming::{PageHinkleyChange, PageHinkleyDetector};
use driftwatch::{
    EqualFrequencyBinning, OnlineDistribution, ReferenceDistribution, StreamingMonitor,
};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn online_psi_approximates_batch_psi() {
    // Reference on [0, 1]; stream the same distribution → online PSI ≈ 0.
    let reference = ReferenceDistribution::fit_continuous(
        "x",
        &linspace(0.0, 1.0, 1000),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap();

    let edges = match reference.histogram().bins() {
        driftwatch::BinDefinition::Continuous { edges } => edges.clone(),
        _ => unreachable!(),
    };

    let mut on_dist = OnlineDistribution::new();
    for v in linspace(0.0, 1.0, 1000) {
        on_dist.update(v);
    }
    let snap = on_dist.snapshot_histogram(&edges).unwrap();
    let psi = driftwatch::psi(reference.histogram(), &snap).unwrap();
    assert!(
        psi < 0.05,
        "same distribution should give near-zero PSI, got {psi}"
    );
}

#[test]
fn streaming_monitor_flags_a_shift() {
    let reference = ReferenceDistribution::fit_continuous(
        "latency",
        &linspace(0.0, 1.0, 1000),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap();

    let mut monitor = StreamingMonitor::new();
    monitor.add_feature(&reference, 0.25).unwrap();

    // Stream values shifted well above the reference range.
    for v in linspace(2.0, 3.0, 1000) {
        monitor.update("latency", v).unwrap();
    }
    let report = monitor.report().unwrap();
    assert_eq!(report.features.len(), 1);
    assert!(
        report.features[0].drifted,
        "online PSI: {}",
        report.features[0].psi
    );
    assert!(report.dataset_drift_detected());
}

#[test]
fn streaming_monitor_rejects_categorical() {
    let cat = ReferenceDistribution::fit_categorical("c", &["a", "b"]).unwrap();
    let mut monitor = StreamingMonitor::new();
    assert!(monitor.add_feature(&cat, 0.25).is_err());
}

#[test]
fn page_hinkley_detects_upward_shift_only_after_it_happens() {
    let mut ph = PageHinkleyDetector::new(0.5, 50.0);
    let mut detected_at = None;
    for i in 0..2000usize {
        let x = if i < 1000 { 0.0 } else { 5.0 };
        if let Some(change) = ph.update(x) {
            detected_at = Some((i, change));
            break;
        }
    }
    let (at, change) = detected_at.expect("should detect the mean shift");
    assert!(at >= 1000, "must not fire before the shift (fired at {at})");
    assert_eq!(change, PageHinkleyChange::Increase);
}

#[test]
fn page_hinkley_stays_quiet_on_stationary_signal() {
    let mut ph = PageHinkleyDetector::new(0.5, 50.0);
    // Deterministic zero-mean oscillation: no change-point.
    for i in 0..5000usize {
        let x = if i % 2 == 0 { 0.3 } else { -0.3 };
        assert!(ph.update(x).is_none(), "false positive at {i}");
    }
}
