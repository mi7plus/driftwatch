# driftwatch

Data & model **drift detection** for Rust — population stability index (PSI),
KL/JS divergence, distributional hypothesis tests (Kolmogorov–Smirnov,
chi-square), live windowing for served models, and a pluggable alerting hook.

`driftwatch` fills a gap in the Rust ML ecosystem: there was no equivalent of
Python's [Evidently](https://github.com/evidentlyai/evidently) or
[WhyLabs](https://whylabs.ai/) for computing how far a *live* data or prediction
distribution has moved from a trusted *reference* (training/baseline)
distribution. This crate computes drift; it deliberately does **not** try to
replace the operational observability tooling (`tracing`, `metrics`,
Prometheus) that already does its job well — it plugs into it.

```rust
use driftwatch::{DatasetMonitor, ReferenceDistribution, EqualFrequencyBinning, LiveFeature};

// Fit a reference distribution per feature from baseline data.
let baseline: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
let reference =
    ReferenceDistribution::fit_continuous("score", &baseline, EqualFrequencyBinning::default())
        .unwrap();

let mut monitor = DatasetMonitor::new();
monitor.add_feature(reference);

// Check a live batch that has shifted upward.
let live: Vec<f64> = (0..1000).map(|i| 0.5 + i as f64 / 1000.0).collect();
let report = monitor.check(&[("score", LiveFeature::Continuous(&live))]).unwrap();
println!("{report}");
assert!(report.features[0].drifted());
```

## What's covered

| Capability                                   | driftwatch | Evidently / WhyLabs |
|----------------------------------------------|:----------:|:-------------------:|
| PSI (population stability index)             | ✅         | ✅                  |
| KL / JS divergence                           | ✅         | ✅                  |
| KS test / chi-square test (with p-values)    | ✅         | ✅                  |
| Per-feature + aggregate dataset-drift report | ✅         | ✅                  |
| Prediction drift preset                      | ✅         | ✅                  |
| Label / concept drift (rolling score)        | ✅         | ✅                  |
| Live windowing for served models             | ✅         | ✅                  |
| Pluggable alerting (log / webhook / custom)  | ✅         | ✅                  |
| Prometheus / `metrics` export                | ✅         | partial             |
| **Live dashboard UI** (self-refreshing web page) | ✅ (`dashboard` feature) | ✅ (signature feature) |
| **Static exportable HTML report file**       | ✅         | ✅                  |
| **Data-quality profiling** (missing / schema) | ✅ (drift-focused) | ✅              |
| **Streaming / online divergence estimation** | ✅ (`streaming` feature, approximate) | partial |

### Still out of scope

These are named plainly rather than left for you to discover by surprise:

- **No hosted, multi-user dashboard service.** The `dashboard` feature is a
  single-process live UI you serve yourself — not a persistent, authenticated,
  multi-tenant monitoring platform.
- **No exact online two-sample tests.** Streaming drift reconstructs the live
  distribution from a quantile sketch, so online PSI/KL/JS are *approximate*
  (they converge to the batch values within sketch accuracy). Exact KS /
  chi-square remain windowed, batch operations.
- **Data-quality profiling is drift-adjacent, not a full data-validation suite.**
  It covers missing-value rates, basic per-feature stats, and schema conformance
  (range / category / kind) — not arbitrary business-rule validation.

## Metrics at a glance

- **PSI** — the credit-risk workhorse. Threshold *convention* (not a
  mathematical fact): `< 0.1` no significant change, `0.1–0.25` moderate, `> 0.25`
  significant.
- **KL divergence** — directional (`D_KL(live ‖ reference)` by default; it is
  asymmetric, so the direction is documented).
- **JS divergence** — symmetric and bounded to `[0, ln 2]`; the one to reach for
  when comparing a single bounded score across features.
- **KS test** — two-sample, on *raw* samples, so it is insensitive to the
  binning-strategy choice the binned metrics depend on.
- **Chi-square test** — a homogeneity test for categorical features, adding a
  significance judgment to PSI's magnitude-only signal.

Both binning strategies are provided; **equal-frequency (quantile) binning is
the default convention for PSI** and what you should usually reach for.
Zero-frequency bins (including a *novel category* that appears only in live data
— itself a strong drift signal) are handled by epsilon smoothing so the
divergence metrics never return `inf`/`NaN`.

## Feature flags

All optional integrations are off by default, keeping the core crate dependency-light.

| Feature             | Pulls in           | Adds                                             |
|---------------------|--------------------|-------------------------------------------------|
| `alerting-log`      | `tracing`          | `LogAlerter` — structured `tracing` event       |
| `alerting-webhook`  | `reqwest`, `serde` | `WebhookAlerter` — POST JSON to any URL          |
| `prometheus-export` | `metrics`          | drift-score gauges on your existing `/metrics`   |
| `label-drift`       | `model-selection-rs` | `LabelDriftMonitor` over its `Scorer` trait    |
| `dashboard`         | `axum`, `tokio`, `serde` | `Dashboard` — a live self-refreshing web UI + JSON API |
| `streaming`         | `sketches-ddsketch` | `OnlineDistribution`, `StreamingMonitor`, `PageHinkleyDetector` |

Data-quality profiling (`DatasetProfile`, `Schema`) and the static `HtmlReport`
are part of the **core** crate — no feature flag, no extra dependencies.

Custom alerting to any other destination (Slack, PagerDuty, a database) is a
matter of implementing the one-method `Alerter` trait yourself.

## Live dashboard

Enable the `dashboard` feature for a self-contained, auto-refreshing web UI that
shows the latest per-feature verdicts, scores, and a dataset-drift banner. You
own the cadence — keep calling `check` and hand each report to the dashboard:

```rust,ignore
use driftwatch::Dashboard;

let dashboard = Dashboard::with_title("Payments model");

// From your serving loop, after each DatasetMonitor::check:
// dashboard.update(report);

// Serve it standalone…
dashboard.serve("127.0.0.1:8080".parse().unwrap()).await?;
// …or mount it into an existing axum app:
//   Router::new().nest("/drift", dashboard.router())
```

The page is a single HTML document with inlined CSS/JS and no external assets; it
polls `/api/report` (JSON) every two seconds. The dashboard renders drift — it
does not compute or schedule it, and it runs no hosted/multi-user infrastructure.

## Static HTML report

For a "save this run" artifact — attach it to a CI job, email it — render a
`DriftReport` to a standalone HTML file with inline-SVG reference-vs-live
histograms. Dependency-free (no plotting stack), no external assets:

```rust,ignore
use driftwatch::HtmlReport;

HtmlReport::new(&report).with_title("Nightly drift").save("report.html")?;
```

## Data-quality profiling

Drift asks "has the distribution moved?"; profiling asks "is this batch even
well-formed?". Derive a `Schema` from your reference distributions and validate
each live batch — missing/unexpected features, kind mismatches, out-of-range
values, novel categories, and null-rate breaches:

```rust,ignore
use driftwatch::{Schema, DatasetProfile, LiveFeature};

let schema = Schema::from_references(&[reference_a, reference_b]);
let report = schema.validate(&[("age", LiveFeature::Continuous(&ages))]);
assert!(report.is_valid());

let mut profile = DatasetProfile::new();
profile.profile_continuous("age", &ages);  // count, missing rate, min/max/mean/std
```

Non-finite values are counted as missing rather than erroring.

## Streaming / online drift

Enable `streaming` to absorb values one at a time into a quantile sketch and
query drift at any instant — no window to buffer or re-bin — plus a Page-Hinkley
change-point detector for scalar signals:

```rust,ignore
use driftwatch::{StreamingMonitor, streaming::PageHinkleyDetector};

let mut monitor = StreamingMonitor::new();
monitor.add_feature(&reference, 0.25)?;   // PSI threshold
monitor.update("latency", value)?;        // per request, O(1)-ish, bounded memory
let report = monitor.report()?;           // online PSI right now

let mut ph = PageHinkleyDetector::new(0.5, 50.0);
if let Some(change) = ph.update(z_score) { /* mean shifted */ }
```

Online PSI/KL/JS are approximate (reconstructed from the sketch); they converge
to the batch values within sketch accuracy.

## Examples

- [`basic_drift_check`](examples/basic_drift_check.rs) — reference vs. drifted
  live batch, printed report, `LogAlerter` firing.
- [`axum_integration`](examples/axum_integration.rs) — a `LiveWindow` fed by a
  request handler, a `tokio::time::interval` loop driving periodic checks, drift
  gauges on `/metrics`.
- [`label_drift`](examples/label_drift.rs) — a simulated degrading-accuracy
  scenario using the `label-drift` feature.
- [`live_dashboard`](examples/live_dashboard.rs) — a simulated serving system
  whose data drifts over time, served on a live dashboard at `127.0.0.1:8080`
  (`--features dashboard`).
- [`html_report`](examples/html_report.rs) — render a standalone `report.html`
  with reference-vs-live distribution charts.
- [`data_quality`](examples/data_quality.rs) — profile a batch and validate it
  against a schema derived from the reference data.
- [`streaming_drift`](examples/streaming_drift.rs) — online PSI and a
  Page-Hinkley change-point detector over a drifting stream (`--features streaming`).

## MSRV & license

MSRV 1.75. Licensed under the [MIT license](LICENSE).
