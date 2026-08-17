//! With the `prometheus-export` feature on, a check must emit correctly-labeled
//! gauges onto the installed `metrics` recorder.
#![cfg(feature = "prometheus-export")]

use driftwatch::export::{DATASET_DRIFT_GAUGE, FEATURE_DRIFT_GAUGE};
use driftwatch::{DatasetMonitor, EqualFrequencyBinning, LiveFeature, ReferenceDistribution};
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn check_emits_drift_gauges() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().expect("install debugging recorder");

    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "score",
            &linspace(0.0, 1.0, 500),
            EqualFrequencyBinning::new(10).unwrap(),
        )
        .unwrap(),
    );

    let shifted = linspace(2.0, 3.0, 500);
    let report = monitor
        .check(&[("score", LiveFeature::Continuous(&shifted))])
        .unwrap();
    assert!(report.dataset_drift_detected());

    let snapshot = snapshotter.snapshot().into_vec();

    let mut dataset_gauge = None;
    let mut feature_gauge = None;
    for (composite, _unit, _desc, value) in snapshot {
        let key = composite.key();
        let name = key.name();
        if let DebugValue::Gauge(v) = value {
            if name == DATASET_DRIFT_GAUGE {
                dataset_gauge = Some(v.into_inner());
            } else if name == FEATURE_DRIFT_GAUGE {
                let has_feature_label = key
                    .labels()
                    .any(|l| l.key() == "feature" && l.value() == "score");
                assert!(has_feature_label, "feature gauge missing 'feature=score'");
                feature_gauge = Some(v.into_inner());
            }
        }
    }

    assert_eq!(
        dataset_gauge,
        Some(1.0),
        "dataset drift gauge should be 1.0"
    );
    let feature_gauge = feature_gauge.expect("feature drift gauge should be present");
    assert!(
        feature_gauge > 0.25,
        "feature drift score should exceed threshold"
    );
}
