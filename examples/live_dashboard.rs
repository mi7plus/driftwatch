//! A live drift dashboard: a background task simulates a serving system whose
//! feature distributions drift over time, runs a `DatasetMonitor::check` on a
//! cadence, and publishes each report to a `Dashboard` served over HTTP.
//!
//! Run it, then open <http://127.0.0.1:8080> in a browser and watch the verdict
//! flip from green to red as the simulated data drifts. Runs until Ctrl-C.
//!
//! ```text
//! cargo run --example live_dashboard --features dashboard
//! ```

use driftwatch::{
    Dashboard, DatasetMonitor, EqualFrequencyBinning, LiveFeature, ReferenceDistribution,
};
use std::sync::Arc;
use std::time::Duration;

/// A tiny deterministic PRNG so the example needs no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sample(&mut self, mean: f64, spread: f64) -> f64 {
        let u = self.unit() + self.unit() + self.unit();
        mean + (u - 1.5) * spread
    }
}

#[tokio::main]
async fn main() {
    let mut rng = Lcg(0xC0FFEE);

    // Reference distributions for two features.
    let ref_latency: Vec<f64> = (0..2000).map(|_| rng.sample(100.0, 20.0)).collect();
    let ref_amount: Vec<f64> = (0..2000).map(|_| rng.sample(50.0, 10.0)).collect();

    // Flag the dataset once *any* of the two features drifts (40% threshold),
    // so the banner visibly flips to red as the latency distribution shifts.
    let mut monitor = DatasetMonitor::new().with_dataset_fraction_threshold(0.4);
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
    let monitor = Arc::new(monitor);

    let dashboard = Dashboard::with_title("Payments model");

    // Background "serving system": every second, generate a live batch whose
    // latency drifts progressively upward, check it, and publish the report.
    {
        let dashboard = dashboard.clone();
        let monitor = Arc::clone(&monitor);
        tokio::spawn(async move {
            let mut step = 0u32;
            loop {
                // latency mean creeps from 100 → ~200 over 30 steps.
                let latency_mean = 100.0 + (step.min(30) as f64) * 3.3;
                let live_latency: Vec<f64> =
                    (0..500).map(|_| rng.sample(latency_mean, 22.0)).collect();
                let live_amount: Vec<f64> = (0..500).map(|_| rng.sample(50.0, 10.0)).collect();

                if let Ok(report) = monitor.check(&[
                    ("latency_ms", LiveFeature::Continuous(&live_latency)),
                    ("amount", LiveFeature::Continuous(&live_amount)),
                ]) {
                    dashboard.update(report);
                }
                step += 1;
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    let addr = "127.0.0.1:8080".parse().unwrap();
    println!("driftwatch dashboard live at http://{addr}  (Ctrl-C to stop)");
    dashboard.serve(addr).await.unwrap();
}
