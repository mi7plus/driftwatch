//! Data-quality profiling and schema validation.

use approx::assert_abs_diff_eq;
use driftwatch::profile::{DatasetProfile, FeatureProfile};
use driftwatch::{
    ContinuousProfile, EqualFrequencyBinning, LiveFeature, ReferenceDistribution, Schema,
    ValidationIssue,
};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

#[test]
fn continuous_profile_counts_non_finite_as_missing() {
    let data = [1.0, 2.0, 3.0, f64::NAN, 5.0, f64::INFINITY];
    let p = ContinuousProfile::compute(&data);
    assert_eq!(p.count, 4);
    assert_eq!(p.missing, 2);
    assert_abs_diff_eq!(p.min, 1.0);
    assert_abs_diff_eq!(p.max, 5.0);
    assert_abs_diff_eq!(p.mean, (1.0 + 2.0 + 3.0 + 5.0) / 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(p.missing_rate(), 2.0 / 6.0, epsilon = 1e-12);
}

#[test]
fn all_missing_continuous_is_nan_not_panic() {
    let data = [f64::NAN, f64::NAN];
    let p = ContinuousProfile::compute(&data);
    assert_eq!(p.count, 0);
    assert_eq!(p.missing, 2);
    assert!(p.mean.is_nan());
    assert_abs_diff_eq!(p.missing_rate(), 1.0);
}

#[test]
fn categorical_profile_top_and_missing() {
    let data = ["a", "a", "a", "b", "", "c", ""];
    let profile = DatasetProfile::new()
        .profile_categorical("color", &data)
        .get("color")
        .cloned()
        .unwrap();
    match profile {
        FeatureProfile::Categorical(p) => {
            assert_eq!(p.count, 5);
            assert_eq!(p.missing, 2);
            assert_eq!(p.cardinality, 3);
            assert_eq!(p.top[0], ("a".to_string(), 3));
        }
        _ => panic!("expected categorical"),
    }
}

fn schema() -> Schema {
    let cont = ReferenceDistribution::fit_continuous(
        "score",
        &linspace(0.0, 1.0, 200),
        EqualFrequencyBinning::new(10).unwrap(),
    )
    .unwrap();
    let cat = ReferenceDistribution::fit_categorical("color", &["red", "green", "blue"]).unwrap();
    Schema::from_references(&[cont, cat])
}

#[test]
fn clean_batch_validates() {
    let scores = linspace(0.1, 0.9, 50);
    let colors = ["red", "green", "blue", "red"];
    let report = schema().validate(&[
        ("score", LiveFeature::Continuous(&scores)),
        ("color", LiveFeature::Categorical(&colors)),
    ]);
    assert!(report.is_valid(), "unexpected issues: {report}");
}

#[test]
fn detects_missing_and_unexpected_features() {
    let extra = linspace(0.1, 0.9, 10);
    // "score" and "color" both missing; "surprise" is unexpected.
    let report = schema().validate(&[("surprise", LiveFeature::Continuous(&extra))]);
    assert!(report.issues.iter().any(
        |i| matches!(i, ValidationIssue::UnexpectedFeature { feature } if feature == "surprise")
    ));
    assert!(report
        .issues
        .iter()
        .any(|i| matches!(i, ValidationIssue::MissingFeature { feature } if feature == "score")));
    assert!(report
        .issues
        .iter()
        .any(|i| matches!(i, ValidationIssue::MissingFeature { feature } if feature == "color")));
}

#[test]
fn detects_out_of_range_and_novel_categories() {
    let scores = [0.5, 0.6, 9.0, -3.0]; // two out of [0,1]
    let colors = ["red", "yellow", "green"]; // yellow is novel
    let report = schema().validate(&[
        ("score", LiveFeature::Continuous(&scores)),
        ("color", LiveFeature::Categorical(&colors)),
    ]);
    assert!(report.issues.iter().any(|i| matches!(
        i,
        ValidationIssue::OutOfRange { feature, count: 2, .. } if feature == "score"
    )));
    assert!(report.issues.iter().any(|i| matches!(
        i,
        ValidationIssue::NovelCategories { feature, categories } if feature == "color" && categories == &["yellow".to_string()]
    )));
}

#[test]
fn detects_kind_mismatch() {
    let colors_as_continuous = [1.0, 2.0, 3.0];
    let report = schema().validate(&[("color", LiveFeature::Continuous(&colors_as_continuous))]);
    assert!(report
        .issues
        .iter()
        .any(|i| matches!(i, ValidationIssue::KindMismatch { feature, .. } if feature == "color")));
}

#[test]
fn null_rate_gate_is_configurable() {
    let scores = [0.5, f64::NAN, 0.7, 0.8]; // 25% missing
    let colors = ["red", "green", "blue"];

    // Strict default (0.0) flags any missing.
    let strict = schema().validate(&[
        ("score", LiveFeature::Continuous(&scores)),
        ("color", LiveFeature::Categorical(&colors)),
    ]);
    assert!(strict
        .issues
        .iter()
        .any(|i| matches!(i, ValidationIssue::HighNullRate { feature, .. } if feature == "score")));

    // Tolerating 50% nulls clears it.
    let lenient = schema().validate_with(
        &[
            ("score", LiveFeature::Continuous(&scores)),
            ("color", LiveFeature::Categorical(&colors)),
        ],
        0.5,
    );
    assert!(lenient.is_valid(), "unexpected: {lenient}");
}
