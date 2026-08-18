//! Schema derivation and validation.

use super::{CategoricalProfile, ContinuousProfile};
use crate::binning::BinDefinition;
use crate::distribution::{FeatureKind, LiveFeature, ReferenceDistribution};
use std::fmt;

/// Default null-rate above which a feature is flagged during validation.
///
/// The default is `0.0` — a strict gate where *any* missing value is reported.
/// Loosen it with [`Schema::validate_with`].
pub const DEFAULT_MAX_NULL_RATE: f64 = 0.0;

/// The expected shape of a single feature.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureSchema {
    /// Feature name.
    pub name: String,
    /// Whether the feature is continuous or categorical.
    pub kind: FeatureKind,
    /// Expected `[min, max]` range for a continuous feature (the reference's
    /// observed range).
    pub range: Option<(f64, f64)>,
    /// Known categories for a categorical feature (the reference's observed set).
    pub categories: Option<Vec<String>>,
}

/// The expected shape of a whole dataset, derived from reference distributions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Schema {
    /// Expected features.
    pub features: Vec<FeatureSchema>,
}

impl Schema {
    /// Derive a schema from a set of fitted [`ReferenceDistribution`]s: each
    /// feature's kind, its continuous range (from the reference bin edges), or
    /// its categorical set.
    pub fn from_references(references: &[ReferenceDistribution]) -> Self {
        let features = references
            .iter()
            .map(|r| {
                let (range, categories) = match r.histogram().bins() {
                    BinDefinition::Continuous { edges } => {
                        let range = if edges.len() >= 2 {
                            Some((edges[0], edges[edges.len() - 1]))
                        } else {
                            None
                        };
                        (range, None)
                    }
                    BinDefinition::Categorical { categories } => (None, Some(categories.clone())),
                };
                FeatureSchema {
                    name: r.name().to_string(),
                    kind: r.kind(),
                    range,
                    categories,
                }
            })
            .collect();
        Self { features }
    }

    /// Validate a live batch against this schema with the default null-rate gate
    /// ([`DEFAULT_MAX_NULL_RATE`]).
    pub fn validate(&self, batch: &[(&str, LiveFeature)]) -> ValidationReport {
        self.validate_with(batch, DEFAULT_MAX_NULL_RATE)
    }

    /// Validate a live batch, flagging any feature whose missing-value rate
    /// exceeds `max_null_rate`.
    pub fn validate_with(
        &self,
        batch: &[(&str, LiveFeature)],
        max_null_rate: f64,
    ) -> ValidationReport {
        let mut issues = Vec::new();

        // Unexpected features and per-feature checks.
        for &(name, feature) in batch {
            let Some(schema) = self.features.iter().find(|s| s.name == name) else {
                issues.push(ValidationIssue::UnexpectedFeature {
                    feature: name.to_string(),
                });
                continue;
            };
            check_feature(schema, name, feature, max_null_rate, &mut issues);
        }

        // Expected features that never showed up.
        for schema in &self.features {
            if !batch.iter().any(|(n, _)| *n == schema.name) {
                issues.push(ValidationIssue::MissingFeature {
                    feature: schema.name.clone(),
                });
            }
        }

        ValidationReport { issues }
    }
}

fn check_feature(
    schema: &FeatureSchema,
    name: &str,
    feature: LiveFeature,
    max_null_rate: f64,
    issues: &mut Vec<ValidationIssue>,
) {
    match (schema.kind, feature) {
        (FeatureKind::Continuous, LiveFeature::Continuous(data)) => {
            let profile = ContinuousProfile::compute(data);
            if profile.missing_rate() > max_null_rate {
                issues.push(ValidationIssue::HighNullRate {
                    feature: name.to_string(),
                    rate: profile.missing_rate(),
                    threshold: max_null_rate,
                });
            }
            if let Some((lo, hi)) = schema.range {
                let out = data
                    .iter()
                    .filter(|v| v.is_finite() && (**v < lo || **v > hi))
                    .count();
                if out > 0 {
                    issues.push(ValidationIssue::OutOfRange {
                        feature: name.to_string(),
                        count: out,
                        observed_min: profile.min,
                        observed_max: profile.max,
                        expected: (lo, hi),
                    });
                }
            }
        }
        (FeatureKind::Categorical, LiveFeature::Categorical(data)) => {
            let profile = CategoricalProfile::compute(data);
            if profile.missing_rate() > max_null_rate {
                issues.push(ValidationIssue::HighNullRate {
                    feature: name.to_string(),
                    rate: profile.missing_rate(),
                    threshold: max_null_rate,
                });
            }
            if let Some(known) = &schema.categories {
                let mut novel: Vec<String> = data
                    .iter()
                    .filter(|c| !c.is_empty() && !known.iter().any(|k| k == *c))
                    .map(|c| c.to_string())
                    .collect();
                novel.sort();
                novel.dedup();
                if !novel.is_empty() {
                    issues.push(ValidationIssue::NovelCategories {
                        feature: name.to_string(),
                        categories: novel,
                    });
                }
            }
        }
        (expected, got) => {
            issues.push(ValidationIssue::KindMismatch {
                feature: name.to_string(),
                expected,
                got: match got {
                    LiveFeature::Continuous(_) => FeatureKind::Continuous,
                    LiveFeature::Categorical(_) => FeatureKind::Categorical,
                },
            });
        }
    }
}

/// A single schema-conformance problem found during validation.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationIssue {
    /// A feature the schema expects was absent from the batch.
    MissingFeature {
        /// The absent feature's name.
        feature: String,
    },
    /// A feature in the batch is not part of the schema.
    UnexpectedFeature {
        /// The unexpected feature's name.
        feature: String,
    },
    /// A feature was supplied with the wrong kind (continuous vs categorical).
    KindMismatch {
        /// Feature name.
        feature: String,
        /// The kind the schema expects.
        expected: FeatureKind,
        /// The kind actually supplied.
        got: FeatureKind,
    },
    /// A continuous feature had values outside its expected range.
    OutOfRange {
        /// Feature name.
        feature: String,
        /// How many values fell outside the range.
        count: usize,
        /// Minimum value observed in the batch.
        observed_min: f64,
        /// Maximum value observed in the batch.
        observed_max: f64,
        /// The `[min, max]` range the schema expects.
        expected: (f64, f64),
    },
    /// A categorical feature contained categories not in the known set.
    NovelCategories {
        /// Feature name.
        feature: String,
        /// The novel categories, sorted and de-duplicated.
        categories: Vec<String>,
    },
    /// A feature's missing-value rate exceeded the configured threshold.
    HighNullRate {
        /// Feature name.
        feature: String,
        /// Observed missing rate.
        rate: f64,
        /// The threshold that was exceeded.
        threshold: f64,
    },
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationIssue::MissingFeature { feature } => {
                write!(f, "missing feature '{feature}'")
            }
            ValidationIssue::UnexpectedFeature { feature } => {
                write!(f, "unexpected feature '{feature}' not in schema")
            }
            ValidationIssue::KindMismatch {
                feature,
                expected,
                got,
            } => write!(
                f,
                "feature '{feature}' has wrong kind: expected {expected:?}, got {got:?}"
            ),
            ValidationIssue::OutOfRange {
                feature,
                count,
                observed_min,
                observed_max,
                expected,
            } => write!(
                f,
                "feature '{feature}': {count} value(s) outside expected range [{:.4}, {:.4}] (observed [{:.4}, {:.4}])",
                expected.0, expected.1, observed_min, observed_max
            ),
            ValidationIssue::NovelCategories { feature, categories } => {
                write!(f, "feature '{feature}': novel categories {categories:?}")
            }
            ValidationIssue::HighNullRate {
                feature,
                rate,
                threshold,
            } => write!(
                f,
                "feature '{feature}': null rate {:.1}% exceeds threshold {:.1}%",
                rate * 100.0,
                threshold * 100.0
            ),
        }
    }
}

/// The outcome of validating a batch against a [`Schema`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationReport {
    /// Every conformance problem found, in the order detected.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Whether the batch conformed to the schema (no issues).
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            return write!(f, "schema OK (no issues)");
        }
        writeln!(f, "schema validation: {} issue(s)", self.issues.len())?;
        for issue in &self.issues {
            writeln!(f, "  - {issue}")?;
        }
        Ok(())
    }
}
