//! Online/streaming drift: absorb values one at a time and query drift at any
//! instant, plus a Page-Hinkley change-point detector on a scalar signal.
//!
//! ```text
//! cargo run --example streaming_drift --features streaming
//! ```

use driftwatch::streaming::PageHinkleyDetector;
use driftwatch::{EqualFrequencyBinning, ReferenceDistribution, StreamingMonitor};

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
    let mut rng = Lcg(0xABCDEF);

    let reference = ReferenceDistribution::fit_continuous(
        "latency_ms",
        &(0..5000)
            .map(|_| rng.sample(100.0, 20.0))
            .collect::<Vec<_>>(),
        EqualFrequencyBinning::default(),
    )
    .unwrap();

    let mut monitor = StreamingMonitor::new();
    monitor.add_feature(&reference, 0.25).unwrap();
    // Page-Hinkley works best on a standardized, low-variance signal, so we feed
    // it the z-score of each latency (reference mean 100, sd 20) rather than the
    // raw millisecond value. delta ≈ half a sd, lambda the detection budget.
    let (ref_mean, ref_sd) = (100.0, 20.0);
    let mut ph = PageHinkleyDetector::new(0.5, 50.0);

    println!("step   count   online-PSI   verdict");
    // Stream 6000 requests; latency mean steps up (100 → 160 ms) at step 3000.
    for step in 0..6000 {
        let mean = if step < 3000 { 100.0 } else { 160.0 };
        let value = rng.sample(mean, 20.0);
        monitor.update("latency_ms", value).unwrap();

        // The Page-Hinkley detector runs on every value and prints when it fires.
        if let Some(change) = ph.update((value - ref_mean) / ref_sd) {
            println!(
                "  ↳ step {}: Page-Hinkley change-point ({change:?})",
                step + 1
            );
        }

        if step % 1000 == 999 {
            let report = monitor.report().unwrap();
            let f = &report.features[0];
            println!(
                "{:>5}  {:>6}   {:>9.4}   {:>7}",
                step + 1,
                f.count,
                f.psi,
                if f.drifted { "DRIFT" } else { "ok" },
            );
        }
    }
}
