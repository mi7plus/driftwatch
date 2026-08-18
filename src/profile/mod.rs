//! Data-quality profiling and schema validation.
//!
//! These are quality checks that run *alongside* drift, not a replacement for
//! it: missing-value rates, basic per-feature statistics, and conformance of a
//! live batch to a schema derived from the reference data. Drift asks "has the
//! distribution moved?"; profiling asks "is this batch even well-formed?" — a
//! batch can be perfectly on-distribution and still be 40% nulls, or arrive with
//! a feature missing entirely.
//!
//! # Missing values
//!
//! For continuous features, a **non-finite value (`NaN`/`±inf`) is treated as
//! missing** — this is the one place the crate accepts non-finite input rather
//! than erroring, precisely so it can *count* it. For categorical features, an
//! **empty string is treated as missing**. Everything else is a present value.

mod schema;

pub use schema::{FeatureSchema, Schema, ValidationIssue, ValidationReport, DEFAULT_MAX_NULL_RATE};

use std::collections::BTreeMap;
use std::fmt;

/// Default number of top categories retained in a [`CategoricalProfile`].
pub const DEFAULT_TOP_K: usize = 10;

/// Summary statistics for a continuous feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousProfile {
    /// Number of present (finite) values.
    pub count: usize,
    /// Number of missing (`NaN`/`±inf`) values.
    pub missing: usize,
    /// Minimum present value (`NaN` if every value was missing).
    pub min: f64,
    /// Maximum present value (`NaN` if every value was missing).
    pub max: f64,
    /// Mean of present values (`NaN` if every value was missing).
    pub mean: f64,
    /// Population standard deviation of present values (`NaN` if empty).
    pub std: f64,
}

impl ContinuousProfile {
    /// Profile a continuous batch, treating non-finite values as missing.
    pub fn compute(data: &[f64]) -> Self {
        let mut count = 0usize;
        let mut missing = 0usize;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for &v in data {
            if v.is_finite() {
                count += 1;
                min = min.min(v);
                max = max.max(v);
                sum += v;
            } else {
                missing += 1;
            }
        }
        if count == 0 {
            return Self {
                count: 0,
                missing,
                min: f64::NAN,
                max: f64::NAN,
                mean: f64::NAN,
                std: f64::NAN,
            };
        }
        let mean = sum / count as f64;
        let var = data
            .iter()
            .filter(|v| v.is_finite())
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        Self {
            count,
            missing,
            min,
            max,
            mean,
            std: var.sqrt(),
        }
    }

    /// Fraction of values that were missing, in `[0, 1]`.
    pub fn missing_rate(&self) -> f64 {
        let total = self.count + self.missing;
        if total == 0 {
            0.0
        } else {
            self.missing as f64 / total as f64
        }
    }
}

/// Summary statistics for a categorical feature.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalProfile {
    /// Number of present (non-empty) labels.
    pub count: usize,
    /// Number of missing (empty-string) labels.
    pub missing: usize,
    /// Number of distinct present categories.
    pub cardinality: usize,
    /// The most frequent categories, `(label, count)`, descending by count
    /// (ties broken alphabetically), truncated to a top-k.
    pub top: Vec<(String, usize)>,
}

impl CategoricalProfile {
    /// Profile a categorical batch (empty strings are missing), keeping the
    /// [`DEFAULT_TOP_K`] most frequent categories.
    pub fn compute(data: &[&str]) -> Self {
        Self::compute_top_k(data, DEFAULT_TOP_K)
    }

    /// [`CategoricalProfile::compute`] with an explicit top-k.
    pub fn compute_top_k(data: &[&str], k: usize) -> Self {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut missing = 0usize;
        for &label in data {
            if label.is_empty() {
                missing += 1;
            } else {
                *counts.entry(label).or_insert(0) += 1;
            }
        }
        let count = data.len() - missing;
        let cardinality = counts.len();
        // BTreeMap iterates alphabetically; a stable sort by descending count
        // then keeps alphabetical order as the tie-break.
        let mut ranked: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        ranked.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        ranked.truncate(k);
        Self {
            count,
            missing,
            cardinality,
            top: ranked,
        }
    }

    /// Fraction of values that were missing, in `[0, 1]`.
    pub fn missing_rate(&self) -> f64 {
        let total = self.count + self.missing;
        if total == 0 {
            0.0
        } else {
            self.missing as f64 / total as f64
        }
    }
}

/// A per-feature profile, continuous or categorical.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureProfile {
    /// Continuous-feature statistics.
    Continuous(ContinuousProfile),
    /// Categorical-feature statistics.
    Categorical(CategoricalProfile),
}

impl FeatureProfile {
    /// The feature's missing-value rate, regardless of kind.
    pub fn missing_rate(&self) -> f64 {
        match self {
            FeatureProfile::Continuous(p) => p.missing_rate(),
            FeatureProfile::Categorical(p) => p.missing_rate(),
        }
    }
}

/// One named feature's profile within a [`DatasetProfile`].
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureProfileEntry {
    /// Feature name.
    pub name: String,
    /// The computed profile.
    pub profile: FeatureProfile,
}

/// A data-quality profile of a batch: one [`FeatureProfile`] per feature, plus a
/// readable [`fmt::Display`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DatasetProfile {
    /// Per-feature profiles in insertion order.
    pub features: Vec<FeatureProfileEntry>,
}

impl DatasetProfile {
    /// An empty profile to add features to.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a continuous feature's profile.
    pub fn profile_continuous(&mut self, name: impl Into<String>, data: &[f64]) -> &mut Self {
        self.features.push(FeatureProfileEntry {
            name: name.into(),
            profile: FeatureProfile::Continuous(ContinuousProfile::compute(data)),
        });
        self
    }

    /// Add a categorical feature's profile.
    pub fn profile_categorical(&mut self, name: impl Into<String>, data: &[&str]) -> &mut Self {
        self.features.push(FeatureProfileEntry {
            name: name.into(),
            profile: FeatureProfile::Categorical(CategoricalProfile::compute(data)),
        });
        self
    }

    /// Look up a feature's profile by name.
    pub fn get(&self, name: &str) -> Option<&FeatureProfile> {
        self.features
            .iter()
            .find(|f| f.name == name)
            .map(|f| &f.profile)
    }
}

impl fmt::Display for DatasetProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DataProfile ({} features)", self.features.len())?;
        for entry in &self.features {
            match &entry.profile {
                FeatureProfile::Continuous(p) => writeln!(
                    f,
                    "  {:<20} continuous  n={} missing={} ({:.1}%)  min={:.4} max={:.4} mean={:.4} std={:.4}",
                    entry.name,
                    p.count,
                    p.missing,
                    p.missing_rate() * 100.0,
                    p.min,
                    p.max,
                    p.mean,
                    p.std,
                )?,
                FeatureProfile::Categorical(p) => {
                    let top = p
                        .top
                        .iter()
                        .take(3)
                        .map(|(c, n)| format!("{c}={n}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(
                        f,
                        "  {:<20} categorical n={} missing={} ({:.1}%)  cardinality={}  top: {}",
                        entry.name,
                        p.count,
                        p.missing,
                        p.missing_rate() * 100.0,
                        p.cardinality,
                        top,
                    )?
                }
            }
        }
        Ok(())
    }
}
