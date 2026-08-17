//! Error types for the crate.
//!
//! Every fallible operation in the public API returns a [`Result`] whose error
//! variant is [`DriftError`]. Library code never panics on bad input reachable
//! from the public API; all failure modes are surfaced here instead.

use thiserror::Error;

/// Convenient alias for `Result<T, DriftError>`.
pub type Result<T> = std::result::Result<T, DriftError>;

/// Errors produced while fitting reference distributions or computing drift.
#[derive(Debug, Error)]
pub enum DriftError {
    /// A binning strategy or metric was handed an empty sample slice.
    #[error("empty input: {0}")]
    EmptyInput(String),

    /// The requested number of bins was zero, or otherwise unusable.
    #[error("invalid bin count: {0}")]
    InvalidBinCount(usize),

    /// The input contained a `NaN` or infinite value, which cannot be binned.
    #[error("input contained a non-finite value at index {index}")]
    NonFinite {
        /// Position of the offending value in the input slice.
        index: usize,
    },

    /// Two histograms being compared do not have the same number of bins, so a
    /// per-bin metric (PSI/KL/JS/chi-square) is undefined between them.
    #[error("histogram length mismatch: reference has {reference} bins, live has {live}")]
    BinCountMismatch {
        /// Bin count of the reference histogram.
        reference: usize,
        /// Bin count of the live histogram.
        live: usize,
    },

    /// A feature named in the live data has no matching reference distribution,
    /// or vice versa.
    #[error("unknown feature: {0}")]
    UnknownFeature(String),

    /// The sample size is below the documented minimum for a meaningful
    /// distributional test (KS / chi-square), so its result would be low-power.
    #[error("sample too small: {kind} needs at least {minimum} samples, got {actual}")]
    SampleTooSmall {
        /// Which test raised the error, e.g. `"KS test"`.
        kind: &'static str,
        /// Documented minimum sample size.
        minimum: usize,
        /// Number of samples actually supplied.
        actual: usize,
    },

    /// The reference data spans zero width (all-identical continuous values), so
    /// no non-degenerate set of bin edges exists.
    #[error("zero-width range: all reference values are identical ({value}); cannot form continuous bins")]
    ZeroWidthRange {
        /// The single value shared by every reference sample.
        value: f64,
    },

    /// A configuration value was outside its valid range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}
