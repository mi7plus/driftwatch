//! Generate a self-contained static HTML drift report and write it to disk.
//!
//! ```text
//! cargo run --example html_report
//! ```

use driftwatch::{
    DatasetMonitor, EqualFrequencyBinning, HtmlReport, LiveFeature, ReferenceDistribution,
};

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

fn main() {
    let mut rng = Lcg(0x5EED);

    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "latency_ms",
            &(0..2000)
                .map(|_| rng.sample(100.0, 20.0))
                .collect::<Vec<_>>(),
            EqualFrequencyBinning::default(),
        )
        .unwrap(),
    );
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "amount",
            &(0..2000)
                .map(|_| rng.sample(50.0, 10.0))
                .collect::<Vec<_>>(),
            EqualFrequencyBinning::default(),
        )
        .unwrap(),
    );

    // Live batch: latency drifted up, amount unchanged.
    let live_latency: Vec<f64> = (0..800).map(|_| rng.sample(160.0, 25.0)).collect();
    let live_amount: Vec<f64> = (0..800).map(|_| rng.sample(50.0, 10.0)).collect();
    let report = monitor
        .check(&[
            ("latency_ms", LiveFeature::Continuous(&live_latency)),
            ("amount", LiveFeature::Continuous(&live_amount)),
        ])
        .unwrap();

    let path = std::env::temp_dir().join("driftwatch_report.html");
    HtmlReport::new(&report)
        .with_title("Nightly drift report")
        .save(&path)
        .unwrap();
    println!(
        "wrote {} ({} features)",
        path.display(),
        report.features.len()
    );
    println!("open it in a browser to see the reference-vs-live distributions.");
}
