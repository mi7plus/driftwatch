//! Per-feature reference distributions and the live data compared against them.
//!
//! A [`ReferenceDistribution`] is fitted once from baseline/training data and
//! then reused to bin every live window *consistently* — live data is always
//! counted against the reference's own chosen bin edges (continuous) or category
//! set (categorical), never rebinned independently. Rebinning live data on its
//! own terms would silently invalidate every per-bin metric, so that invariant
//! is enforced here rather than left to the caller.

use crate::binning::{histogram_from_edges, BinDefinition, ContinuousBinning, Histogram};
use crate::error::{DriftError, Result};

/// A batch of live values for a single feature, matching a
/// [`ReferenceDistribution`]'s kind.
#[derive(Clone, Copy, Debug)]
pub enum LiveFeature<'a> {
    /// Raw continuous samples, compared against continuous reference bins.
    Continuous(&'a [f64]),
    /// Category labels, compared against a categorical reference's category set.
    Categorical(&'a [&'a str]),
}

/// What kind of feature a [`ReferenceDistribution`] describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    /// A continuous feature binned into numeric ranges.
    Continuous,
    /// A discrete/categorical feature counted per category.
    Categorical,
}

enum Reference {
    Continuous {
        edges: Vec<f64>,
        histogram: Histogram,
        samples: Vec<f64>,
    },
    Categorical {
        categories: Vec<String>,
        counts: Vec<f64>,
        histogram: Histogram,
    },
}

/// The reference (baseline) distribution for one named feature.
pub struct ReferenceDistribution {
    name: String,
    reference: Reference,
}

/// An aligned reference/live comparison for one feature, ready to feed to the
/// drift metrics. Produced internally by the dataset monitor's check path.
pub(crate) struct Comparison<'a> {
    /// Reference histogram, aligned to the same bins/categories as `live_hist`.
    pub reference_hist: Histogram,
    /// Live histogram over the reference's bins/categories.
    pub live_hist: Histogram,
    /// Raw `(reference, live)` samples for the KS test, when continuous.
    pub raw_samples: Option<(&'a [f64], &'a [f64])>,
    /// Whether this is a categorical feature (selects chi-square over KS).
    pub kind: FeatureKind,
}

impl ReferenceDistribution {
    /// Fit a continuous reference distribution from baseline `reference` samples
    /// using the given binning strategy (e.g.
    /// [`EqualFrequencyBinning`](crate::EqualFrequencyBinning)).
    ///
    /// The fitted bin edges and the reference histogram are stored, along with
    /// the raw reference samples (so a KS test against live data is possible).
    ///
    /// # Errors
    /// Propagates binning errors: [`DriftError::EmptyInput`] on empty data,
    /// [`DriftError::NonFinite`], and [`DriftError::ZeroWidthRange`] when every
    /// reference value is identical.
    pub fn fit_continuous(
        name: impl Into<String>,
        reference: &[f64],
        binning: impl ContinuousBinning,
    ) -> Result<Self> {
        let edges = binning.fit_edges(reference)?;
        let histogram = histogram_from_edges(&edges, reference)?;
        Ok(Self {
            name: name.into(),
            reference: Reference::Continuous {
                edges,
                histogram,
                samples: reference.to_vec(),
            },
        })
    }

    /// Fit a categorical reference distribution from baseline category labels.
    ///
    /// The observed categories (sorted for determinism) and their counts become
    /// the reference. A category that later appears only in live data is handled
    /// as a zero-reference-frequency bin — a genuinely new category is itself a
    /// strong drift signal, not an error.
    ///
    /// # Errors
    /// [`DriftError::EmptyInput`] if `reference` is empty.
    pub fn fit_categorical(name: impl Into<String>, reference: &[&str]) -> Result<Self> {
        if reference.is_empty() {
            return Err(DriftError::EmptyInput(
                "categorical reference needs at least one label".into(),
            ));
        }
        let mut categories: Vec<String> = reference.iter().map(|s| s.to_string()).collect();
        categories.sort();
        categories.dedup();

        let counts = count_categories(&categories, reference);
        let histogram = Histogram::new(
            BinDefinition::Categorical {
                categories: categories.clone(),
            },
            counts.clone(),
        )?;
        Ok(Self {
            name: name.into(),
            reference: Reference::Categorical {
                categories,
                counts,
                histogram,
            },
        })
    }

    /// The feature's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this reference is continuous or categorical.
    pub fn kind(&self) -> FeatureKind {
        match self.reference {
            Reference::Continuous { .. } => FeatureKind::Continuous,
            Reference::Categorical { .. } => FeatureKind::Categorical,
        }
    }

    /// The reference histogram (before any categorical realignment).
    pub fn histogram(&self) -> &Histogram {
        match &self.reference {
            Reference::Continuous { histogram, .. } => histogram,
            Reference::Categorical { histogram, .. } => histogram,
        }
    }

    /// Bin a live batch against this reference and return an aligned comparison.
    ///
    /// For continuous features the live data is counted into the reference's
    /// fixed edges. For categorical features the reference and live histograms
    /// are both expanded to the union of their categories, so any novel live
    /// category appears as a zero-reference-count bin.
    ///
    /// # Errors
    /// [`DriftError::UnknownFeature`] if `live`'s kind does not match this
    /// reference's kind, plus any binning error for continuous data.
    pub(crate) fn compare<'a>(&'a self, live: LiveFeature<'a>) -> Result<Comparison<'a>> {
        match (&self.reference, live) {
            (
                Reference::Continuous {
                    edges,
                    histogram,
                    samples,
                },
                LiveFeature::Continuous(live_samples),
            ) => {
                let live_hist = histogram_from_edges(edges, live_samples)?;
                Ok(Comparison {
                    reference_hist: histogram.clone(),
                    live_hist,
                    raw_samples: Some((samples.as_slice(), live_samples)),
                    kind: FeatureKind::Continuous,
                })
            }
            (
                Reference::Categorical {
                    categories, counts, ..
                },
                LiveFeature::Categorical(live_labels),
            ) => {
                let (reference_hist, live_hist) =
                    align_categorical(categories, counts, live_labels)?;
                Ok(Comparison {
                    reference_hist,
                    live_hist,
                    raw_samples: None,
                    kind: FeatureKind::Categorical,
                })
            }
            (Reference::Continuous { .. }, LiveFeature::Categorical(_)) => {
                Err(DriftError::UnknownFeature(format!(
                    "feature '{}' is continuous but was given categorical live data",
                    self.name
                )))
            }
            (Reference::Categorical { .. }, LiveFeature::Continuous(_)) => {
                Err(DriftError::UnknownFeature(format!(
                    "feature '{}' is categorical but was given continuous live data",
                    self.name
                )))
            }
        }
    }
}

/// Count occurrences of each category in `categories` within `data`.
fn count_categories(categories: &[String], data: &[&str]) -> Vec<f64> {
    categories
        .iter()
        .map(|c| data.iter().filter(|&&d| d == c.as_str()).count() as f64)
        .collect()
}

/// Build aligned reference/live categorical histograms over the union of the
/// reference categories and any novel categories seen only in `live`.
///
/// The union is `[reference categories in their fitted order] ++ [novel live
/// categories, sorted]`. Reference counts for fitted categories are carried
/// over; novel categories get a zero reference count, which the epsilon
/// smoothing in the divergence metrics then handles cleanly.
fn align_categorical(
    reference_categories: &[String],
    reference_counts: &[f64],
    live: &[&str],
) -> Result<(Histogram, Histogram)> {
    let mut novel: Vec<String> = live
        .iter()
        .filter(|&&l| !reference_categories.iter().any(|c| c == l))
        .map(|s| s.to_string())
        .collect();
    novel.sort();
    novel.dedup();

    let mut union: Vec<String> = reference_categories.to_vec();
    union.extend(novel.iter().cloned());

    // Reference counts: fitted counts for known categories, 0 for novel ones.
    let mut ref_counts = reference_counts.to_vec();
    ref_counts.extend(std::iter::repeat(0.0).take(novel.len()));

    let live_counts = count_categories(&union, live);

    let bins = BinDefinition::Categorical { categories: union };
    let reference_hist = Histogram::new(bins.clone(), ref_counts)?;
    let live_hist = Histogram::new(bins, live_counts)?;
    Ok((reference_hist, live_hist))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::EqualFrequencyBinning;

    #[test]
    fn continuous_reference_histogram_normalizes() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let reference = ReferenceDistribution::fit_continuous(
            "x",
            &baseline,
            EqualFrequencyBinning::new(10).unwrap(),
        )
        .unwrap();
        let sum: f64 = reference.histogram().frequencies().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert_eq!(reference.kind(), FeatureKind::Continuous);
    }

    #[test]
    fn novel_category_becomes_zero_reference_bin() {
        // Reference sees only "a" and "b".
        let reference = ReferenceDistribution::fit_categorical("color", &["a", "a", "b"]).unwrap();
        // Live introduces a genuinely new category "c".
        let live = ["a", "c", "c"];
        let comparison = reference.compare(LiveFeature::Categorical(&live)).unwrap();

        // The aligned category set is the union, with "c" appended.
        if let BinDefinition::Categorical { categories } = comparison.reference_hist.bins() {
            assert_eq!(
                categories,
                &["a".to_string(), "b".to_string(), "c".to_string()]
            );
        } else {
            panic!("expected categorical bins");
        }

        // Reference count for the novel category is exactly zero.
        let ref_counts = comparison.reference_hist.counts();
        assert_eq!(ref_counts, &[2.0, 1.0, 0.0]);
        // Live counts: one "a", zero "b", two "c".
        assert_eq!(comparison.live_hist.counts(), &[1.0, 0.0, 2.0]);

        // Crucially, PSI over this pair is finite despite the zero-reference bin.
        let psi = crate::psi(&comparison.reference_hist, &comparison.live_hist).unwrap();
        assert!(psi.is_finite());
    }

    #[test]
    fn kind_mismatch_is_rejected() {
        let reference = ReferenceDistribution::fit_categorical("c", &["a", "b"]).unwrap();
        let err = reference.compare(LiveFeature::Continuous(&[1.0, 2.0]));
        assert!(err.is_err());
    }
}
