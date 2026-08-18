//! Data-quality profiling and schema validation, alongside drift.
//!
//! ```text
//! cargo run --example data_quality
//! ```

use driftwatch::{
    DatasetProfile, EqualFrequencyBinning, LiveFeature, ReferenceDistribution, Schema,
};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

fn main() {
    // Derive a schema from the reference (training) data.
    let schema = Schema::from_references(&[
        ReferenceDistribution::fit_continuous(
            "age",
            &linspace(18.0, 90.0, 500),
            EqualFrequencyBinning::default(),
        )
        .unwrap(),
        ReferenceDistribution::fit_categorical("plan", &["free", "pro", "enterprise"]).unwrap(),
    ]);

    // A live batch that is subtly broken: some missing ages, an impossible age,
    // and a plan tier that didn't exist at training time.
    let age = [34.0, 41.0, f64::NAN, 205.0, 27.0, 63.0];
    let plan = ["pro", "free", "", "trial", "enterprise"];

    // Profile it.
    let mut profile = DatasetProfile::new();
    profile.profile_continuous("age", &age);
    profile.profile_categorical("plan", &plan);
    println!("{profile}");

    // Validate it against the schema.
    let report = schema.validate(&[
        ("age", LiveFeature::Continuous(&age)),
        ("plan", LiveFeature::Categorical(&plan)),
    ]);
    println!("{report}");
}
