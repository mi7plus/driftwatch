//! Full-pipeline test: reference → monitor (+ alerter) → live batch with
//! injected drift → correct report and correct alerter firing.

use driftwatch::{
    Alerter, DatasetMonitor, DriftAlertEvent, EqualFrequencyBinning, LiveFeature,
    PredictionDriftMonitor, ReferenceDistribution,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Default)]
struct CountingAlerter {
    fired: Arc<AtomicUsize>,
}

impl Alerter for CountingAlerter {
    fn alert(&self, _event: &DriftAlertEvent) {
        self.fired.fetch_add(1, Ordering::SeqCst);
    }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn full_pipeline_flags_injected_drift_and_alerts() {
    let fired = Arc::new(AtomicUsize::new(0));
    let alerter = Box::new(CountingAlerter {
        fired: Arc::clone(&fired),
    });

    let mut monitor = DatasetMonitor::new();
    for name in ["f0", "f1", "f2"] {
        monitor.add_feature(
            ReferenceDistribution::fit_continuous(
                name,
                &linspace(0.0, 1.0, 500),
                EqualFrequencyBinning::new(10).unwrap(),
            )
            .unwrap(),
        );
    }
    let monitor = monitor.with_alerter(alerter);

    // Inject drift on 2 of 3 features → majority → dataset drift → alert fires.
    let unchanged = linspace(0.0, 1.0, 500);
    let shifted = linspace(2.0, 3.0, 500);
    let report = monitor
        .check(&[
            ("f0", LiveFeature::Continuous(&unchanged)),
            ("f1", LiveFeature::Continuous(&shifted)),
            ("f2", LiveFeature::Continuous(&shifted)),
        ])
        .unwrap();

    assert!(report.dataset_drift_detected());
    assert_eq!(report.drifted_features().count(), 2);
    assert_eq!(fired.load(Ordering::SeqCst), 1);
}

#[test]
fn alerter_does_not_fire_without_drift() {
    let fired = Arc::new(AtomicUsize::new(0));
    let monitor = {
        let mut m = DatasetMonitor::new();
        m.add_feature(
            ReferenceDistribution::fit_continuous(
                "f0",
                &linspace(0.0, 1.0, 500),
                EqualFrequencyBinning::new(10).unwrap(),
            )
            .unwrap(),
        );
        m.with_alerter(Box::new(CountingAlerter {
            fired: Arc::clone(&fired),
        }))
    };

    let unchanged = linspace(0.0, 1.0, 500);
    let report = monitor
        .check(&[("f0", LiveFeature::Continuous(&unchanged))])
        .unwrap();
    assert!(!report.dataset_drift_detected());
    assert_eq!(fired.load(Ordering::SeqCst), 0);
}

#[test]
fn prediction_drift_is_a_thin_wrapper() {
    // A PredictionDriftMonitor must produce the same primary score as calling a
    // DatasetMonitor directly on the same data under the feature name it uses.
    let reference = linspace(0.0, 1.0, 500);
    let live = linspace(0.5, 1.5, 500);

    let pred_mon =
        PredictionDriftMonitor::new(&reference, EqualFrequencyBinning::new(10).unwrap()).unwrap();
    let via_wrapper = pred_mon.check(&live).unwrap();

    let mut direct = DatasetMonitor::new();
    direct.add_feature(
        ReferenceDistribution::fit_continuous(
            "prediction",
            &reference,
            EqualFrequencyBinning::new(10).unwrap(),
        )
        .unwrap(),
    );
    let via_direct = direct
        .check(&[("prediction", LiveFeature::Continuous(&live))])
        .unwrap();

    assert_eq!(
        via_wrapper.features[0].primary_statistic(),
        via_direct.features[0].primary_statistic()
    );
    assert_eq!(
        via_wrapper.features[0].drifted(),
        via_direct.features[0].drifted()
    );
}
