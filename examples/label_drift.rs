//! Label / concept drift: track a rolling accuracy over (prediction, actual)
//! pairs and flag when it degrades past a threshold.
//!
//! Run with:
//!
//! ```text
//! cargo run --example label_drift --features label-drift
//! ```

use driftwatch::LabelDriftMonitor;
use model_selection_rs::scoring::Scorer;
use ndarray::Array1;

/// Minimal accuracy scorer implementing model-selection-rs's `Scorer`, so the
/// online metric matches what the offline Evaluation stage would report.
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

/// `n` (prediction, actual) pairs with an `error_rate` fraction wrong.
fn window(n: usize, error_rate: f64) -> Vec<(f64, f64)> {
    let wrong = (n as f64 * error_rate).round() as usize;
    (0..n)
        .map(|i| {
            let actual = 1.0;
            let prediction = if i < wrong { 0.0 } else { 1.0 };
            (prediction, actual)
        })
        .collect()
}

fn main() {
    // Baseline: the model scored 0.95 accuracy on its held-out test set.
    let reference = window(1000, 0.05);
    let monitor = LabelDriftMonitor::from_reference(Accuracy, &reference, 0.10).unwrap();
    println!("baseline accuracy: {:.3}", monitor.baseline_score());
    println!();

    // Simulate concept drift: accuracy erodes over successive live windows as
    // delayed ground-truth labels arrive and are matched to predictions.
    for (day, error_rate) in [0.06, 0.09, 0.15, 0.28].into_iter().enumerate() {
        let live = window(500, error_rate);
        let report = monitor.check(&live).unwrap();
        println!(
            "day {}: accuracy {:.3}  degradation {:+.3}  {}",
            day + 1,
            report.rolling_score,
            report.degradation,
            if report.drifted { "DRIFT" } else { "ok" },
        );
    }
}
