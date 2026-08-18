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
| **Static exportable HTML report file**       | ❌         | ✅                  |
| **Automated data-quality profiling**         | ❌         | ✅                  |
| **Streaming / online divergence estimation** | ❌         | partial             |

### Explicitly out of scope for v1

These are named plainly rather than left for you to discover by surprise:

- **No static exportable HTML report file.** There is a *live* dashboard (see
  below), but not yet a "save this run as a standalone `report.html`" generator.
  Pair `driftwatch`'s structured `DriftReport` with
  [`plotters-statistical`](https://crates.io/crates/plotters-statistical) for
  charts, or read the live dashboard's `/api/report` JSON.
- **No general data-quality profiling** (missing-value rates, schema validation)
  beyond drift specifically.
- **No true streaming / online divergence.** `driftwatch` recomputes drift over
  discrete windows/snapshots, not as a continuously-updated online statistic.

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

## MSRV & license

MSRV 1.75. Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.
