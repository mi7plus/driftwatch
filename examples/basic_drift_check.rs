//! Basic drift check: fit a reference from baseline data, check a deliberately
//! drifted live batch, print the `DriftReport`, and let a `LogAlerter` fire.
//!
//! Run with:
//!
//! ```text
//! cargo run --example basic_drift_check --features alerting-log
//! ```

use driftwatch::{
    DatasetMonitor, EqualFrequencyBinning, LiveFeature, LogAlerter, ReferenceDistribution,
};

/// A tiny deterministic PRNG so the example needs no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f64 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A sample from a normal-ish distribution centered at `mean` (sum of 3
    /// uniforms — crude but dependency-free).
    fn sample(&mut self, mean: f64, spread: f64) -> f64 {
        let u = self.next_unit() + self.next_unit() + self.next_unit();
        mean + (u - 1.5) * spread
    }
}

fn main() {
    // Show `LogAlerter`'s structured events on stdout.
    tracing_subscriber::fmt::init();

    let mut rng = Lcg(0x1234_5678);

    // Baseline / reference data for two features.
    let ref_latency: Vec<f64> = (0..1000).map(|_| rng.sample(100.0, 20.0)).collect();
    let ref_amount: Vec<f64> = (0..1000).map(|_| rng.sample(50.0, 10.0)).collect();

    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "latency_ms",
            &ref_latency,
            EqualFrequencyBinning::default(),
        )
        .unwrap(),
    );
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "amount",
            &ref_amount,
            EqualFrequencyBinning::default(),
        )
        .unwrap(),
    );
    // Flag the dataset once *any* feature drifts (40% threshold), so this
    // one-feature-drifted batch trips the LogAlerter.
    let monitor = monitor
        .with_dataset_fraction_threshold(0.4)
        .with_alerter(Box::new(LogAlerter::new()));

    // Live batch: `latency_ms` has drifted upward, `amount` is unchanged.
    let live_latency: Vec<f64> = (0..500).map(|_| rng.sample(160.0, 25.0)).collect();
    let live_amount: Vec<f64> = (0..500).map(|_| rng.sample(50.0, 10.0)).collect();

    let report = monitor
        .check(&[
            ("latency_ms", LiveFeature::Continuous(&live_latency)),
            ("amount", LiveFeature::Continuous(&live_amount)),
        ])
        .unwrap();

    println!("{report}");
    println!(
        "dataset drift detected: {}",
        report.dataset_drift_detected()
    );
}
