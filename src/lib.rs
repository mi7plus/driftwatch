//! `driftwatch` — data & model drift detection for Rust.
//!
//! `driftwatch` computes how far a *live* data or prediction distribution has
//! moved from a trusted *reference* (training/baseline) distribution, and lets
//! you alert on it. It provides the metrics named in the Rust ML ecosystem gap
//! that Evidently/WhyLabs fill in Python — but no equivalent existed in Rust:
//!
//! * **Binned distribution metrics** — [`psi`], [`kl_divergence`],
//!   [`js_divergence`] over [`Histogram`]s.
//! * **Raw-sample distributional tests** — a two-sample [`ks_test`] and a
//!   categorical [`chi_square_test`], each returning a statistic *and* a
//!   p-value so you apply your own significance threshold.
//! * **Orchestration** — [`ReferenceDistribution`] per feature, a
//!   [`DatasetMonitor`] that turns a live batch into a [`DriftReport`], and a
//!   thread-safe [`LiveWindow`] for feeding it from a serving system.
//! * **Pluggable alerting** — the [`Alerter`] trait, with optional `tracing`
//!   and webhook implementations behind feature flags.
//!
//! # What this crate is *not*
//!
//! It does not render HTML reports or a dashboard UI (pair [`DriftReport`]'s
//! structured data with `plotters-statistical` for charts), it does not do
//! general data-quality profiling beyond drift, and it recomputes drift over
//! discrete windows/snapshots rather than as a continuously-updated online
//! statistic. See the README for the full comparison against Evidently/WhyLabs.
//!
//! # Quick start
//!
//! ```
//! use driftwatch::{DatasetMonitor, ReferenceDistribution, EqualFrequencyBinning, LiveFeature};
//!
//! // Fit a reference distribution per feature from baseline data.
//! let baseline: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
//! let reference = ReferenceDistribution::fit_continuous(
//!     "score",
//!     &baseline,
//!     EqualFrequencyBinning::new(10).unwrap(),
//! )
//! .unwrap();
//!
//! let mut monitor = DatasetMonitor::new();
//! monitor.add_feature(reference);
//!
//! // Check a live batch that has shifted upward.
//! let live: Vec<f64> = (0..200).map(|i| 0.5 + i as f64 / 200.0).collect();
//! let report = monitor.check(&[("score", LiveFeature::Continuous(&live))]).unwrap();
//! println!("{report}");
//! assert!(report.dataset_drift_detected());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod alert;
pub mod binning;
pub mod distribution;
pub mod error;
pub mod metrics;
pub mod monitor;

#[cfg(feature = "prometheus-export")]
pub mod export;

#[cfg(feature = "dashboard")]
pub mod dashboard;

pub use binning::{
    BinDefinition, ContinuousBinning, EqualFrequencyBinning, EqualWidthBinning, Histogram,
    DEFAULT_BIN_COUNT,
};
pub use distribution::{FeatureKind, LiveFeature, ReferenceDistribution};
pub use error::{DriftError, Result};
pub use metrics::{
    chi_square_test, js_divergence, kl_divergence, ks_test, psi, ChiSquareResult, KsTestResult,
    DEFAULT_EPSILON,
};
pub use monitor::{
    DatasetMonitor, DriftReport, DriftVerdict, FeatureConfig, FeatureDrift, LiveWindow, MetricKind,
    MetricScore, PredictionDriftMonitor, WindowMode,
};

#[cfg(feature = "label-drift")]
pub use monitor::{LabelDriftMonitor, LabelDriftReport};

#[cfg(feature = "dashboard")]
pub use dashboard::Dashboard;

pub use alert::{Alerter, DriftAlertEvent, NopAlerter};

#[cfg(feature = "alerting-log")]
pub use alert::LogAlerter;
#[cfg(feature = "alerting-webhook")]
pub use alert::WebhookAlerter;
