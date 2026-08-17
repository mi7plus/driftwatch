//! Label-drift monitoring must flag a degrading-accuracy scenario (feature
//! `label-drift`).
#![cfg(feature = "label-drift")]

use driftwatch::LabelDriftMonitor;
use model_selection_rs::scoring::Scorer;
use ndarray::Array1;

/// Minimal accuracy scorer implementing model-selection-rs's `Scorer`, so the
/// online metric here is the same abstraction the offline Evaluation uses.
struct Accuracy;

impl Scorer for Accuracy {
    fn score(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let n = y_true.len();
        if n == 0 {
            return 0.0;
        }
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| (**a - **b).abs() < 1e-9)
            .count();
        correct as f64 / n as f64
    }

    fn name(&self) -> &str {
        "accuracy"
    }

    fn greater_is_better(&self) -> bool {
        true
    }
}

/// `n` (prediction, actual) pairs where the first `wrong` of them are incorrect.
fn pairs_with_errors(n: usize, wrong: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let actual = 1.0;
            let prediction = if i < wrong { 0.0 } else { 1.0 };
            (prediction, actual)
        })
        .collect()
}

#[test]
fn flags_degrading_accuracy() {
    // Baseline: perfect accuracy on a reference set.
    let reference = pairs_with_errors(100, 0);
    let monitor = LabelDriftMonitor::from_reference(Accuracy, &reference, 0.1).unwrap();
    assert!((monitor.baseline_score() - 1.0).abs() < 1e-9);

    // Live window: 30% wrong → accuracy 0.7 → degradation 0.3 > 0.1 threshold.
    let degraded = pairs_with_errors(100, 30);
    let report = monitor.check(&degraded).unwrap();
    assert!((report.rolling_score - 0.7).abs() < 1e-9);
    assert!((report.degradation - 0.3).abs() < 1e-9);
    assert!(report.drifted);
}

#[test]
fn stable_accuracy_does_not_flag() {
    let reference = pairs_with_errors(100, 0);
    let monitor = LabelDriftMonitor::from_reference(Accuracy, &reference, 0.1).unwrap();

    // 5% wrong → accuracy 0.95 → degradation 0.05 < 0.1 → not drifted.
    let ok = pairs_with_errors(100, 5);
    let report = monitor.check(&ok).unwrap();
    assert!(!report.drifted);
}

#[test]
fn empty_window_is_an_error() {
    let monitor = LabelDriftMonitor::new(Accuracy, 1.0, 0.1);
    assert!(monitor.check(&[]).is_err());
}
