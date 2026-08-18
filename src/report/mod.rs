//! Static, self-contained HTML drift reports.
//!
//! Where the [`dashboard`](crate::dashboard) is a *live* server, this renders a
//! single [`DriftReport`] to a standalone `report.html` you can save, email, or
//! attach to a CI run — the "save this run" artifact Evidently produces with
//! `save_html`. The output is one HTML document with inlined CSS and hand-drawn
//! inline SVG charts: no `plotters`, no external assets, no network fetches.
//!
//! For richer or custom charts, the structured [`DriftReport`] pairs naturally
//! with [`plotters-statistical`](https://crates.io/crates/plotters-statistical);
//! this module covers the common case without pulling in a plotting stack.
//!
//! ```no_run
//! use driftwatch::report::HtmlReport;
//! # fn f(report: &driftwatch::DriftReport) -> std::io::Result<()> {
//! HtmlReport::new(report).with_title("Nightly drift").save("report.html")
//! # }
//! ```

use crate::monitor::{DriftReport, DriftVerdict, FeatureDrift};
use std::fmt::Write as _;
use std::path::Path;

/// Builds a self-contained HTML report from a [`DriftReport`].
pub struct HtmlReport<'a> {
    report: &'a DriftReport,
    title: String,
}

impl<'a> HtmlReport<'a> {
    /// Start a report from a drift report, with a default title.
    pub fn new(report: &'a DriftReport) -> Self {
        Self {
            report,
            title: "driftwatch report".to_string(),
        }
    }

    /// Set the report's title (shown in the page heading and `<title>`).
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Render the report to a self-contained HTML string.
    pub fn to_html(&self) -> String {
        render(self.report, &self.title)
    }

    /// Render and write the report to `path`.
    ///
    /// # Errors
    /// Returns any `std::io::Error` from writing the file.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_html())
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn render(report: &DriftReport, title: &str) -> String {
    let mut body = String::new();

    // Dataset verdict banner.
    let drifted = report.dataset_drift_detected();
    let (banner_class, banner_text) = if drifted {
        (
            "bad",
            format!(
                "Dataset drift detected — {:.0}% of features drifted (threshold {:.0}%)",
                report.drifted_fraction() * 100.0,
                report.dataset_fraction_threshold * 100.0
            ),
        )
    } else {
        (
            "ok",
            format!(
                "No dataset drift — {:.0}% of features drifted (threshold {:.0}%)",
                report.drifted_fraction() * 100.0,
                report.dataset_fraction_threshold * 100.0
            ),
        )
    };
    let _ = write!(
        body,
        r#"<div class="banner {banner_class}"><span class="dot {banner_class}"></span>{}</div>"#,
        escape(&banner_text)
    );

    // Summary bar chart of primary scores.
    body.push_str(&summary_chart(report));

    // Per-feature cards.
    for feature in &report.features {
        body.push_str(&feature_card(feature));
    }

    PAGE.replace("{{TITLE}}", &escape(title))
        .replace("{{BODY}}", &body)
}

/// Horizontal bar chart: each feature's primary score, scaled to the largest
/// score in the report, with a marker at its threshold.
fn summary_chart(report: &DriftReport) -> String {
    if report.features.is_empty() {
        return String::new();
    }
    let max = report
        .features
        .iter()
        .map(|f| f.primary_statistic().max(f.threshold))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);

    let mut rows = String::new();
    for f in &report.features {
        let score = f.primary_statistic();
        let w = (score / max * 100.0).clamp(0.0, 100.0);
        let tick = (f.threshold / max * 100.0).clamp(0.0, 100.0);
        let cls = if f.drifted() { "bad" } else { "ok" };
        let _ = write!(
            rows,
            r#"<div class="sumrow"><div class="sumname">{}</div>
<div class="sumbar"><div class="sumfill {cls}" style="width:{w:.1}%"></div>
<div class="sumtick" style="left:{tick:.1}%" title="threshold"></div></div>
<div class="sumval">{} {:.4}</div></div>"#,
            escape(&f.feature),
            f.primary.label(),
            score
        );
    }
    format!(
        r#"<div class="card"><h2>Primary drift scores</h2>{rows}<div class="legend"><span class="tickmark"></span> threshold</div></div>"#
    )
}

fn feature_card(f: &FeatureDrift) -> String {
    let verdict = match f.verdict {
        DriftVerdict::Drifted => r#"<span class="pill bad">DRIFTED</span>"#,
        DriftVerdict::Stable => r#"<span class="pill ok">stable</span>"#,
    };
    let mut metrics = String::new();
    for s in &f.scores {
        match s.p_value {
            Some(p) => {
                let _ = write!(
                    metrics,
                    "<li>{} = {:.4} <span class=\"muted\">(p = {:.4})</span></li>",
                    s.kind.label(),
                    s.statistic,
                    p
                );
            }
            None => {
                let _ = write!(metrics, "<li>{} = {:.4}</li>", s.kind.label(), s.statistic);
            }
        }
    }

    let svg = histogram_svg(
        &f.reference_histogram.frequencies(),
        &f.live_histogram.frequencies(),
    );

    format!(
        r#"<div class="card"><div class="fhead"><h2>{name} {verdict}</h2>
<div class="muted">{kind} · primary {pm}</div></div>
<div class="fbody"><div class="chart">{svg}
<div class="legend"><span class="sw ref"></span> reference <span class="sw live"></span> live</div></div>
<ul class="metrics">{metrics}</ul></div></div>"#,
        name = escape(&f.feature),
        kind = kind_label(f),
        pm = f.primary.label(),
    )
}

fn kind_label(f: &FeatureDrift) -> &'static str {
    use crate::distribution::FeatureKind;
    match f.kind {
        FeatureKind::Continuous => "continuous",
        FeatureKind::Categorical => "categorical",
    }
}

/// Grouped bar chart overlaying reference and live frequencies, one pair of bars
/// per bin. Pure inline SVG — no plotting dependency.
fn histogram_svg(reference: &[f64], live: &[f64]) -> String {
    let n = reference.len().max(live.len());
    if n == 0 {
        return String::new();
    }
    let (w, h) = (520.0_f64, 150.0_f64);
    let (pad_l, pad_b, pad_t) = (8.0_f64, 14.0_f64, 8.0_f64);
    let plot_w = w - pad_l * 2.0;
    let plot_h = h - pad_b - pad_t;
    let max = reference
        .iter()
        .chain(live)
        .copied()
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);

    let slot = plot_w / n as f64;
    let bar_w = (slot * 0.38).max(1.0);

    let mut bars = String::new();
    for i in 0..n {
        let x0 = pad_l + i as f64 * slot;
        let rf = reference.get(i).copied().unwrap_or(0.0);
        let lf = live.get(i).copied().unwrap_or(0.0);
        let rh = rf / max * plot_h;
        let lh = lf / max * plot_h;
        let ry = pad_t + plot_h - rh;
        let ly = pad_t + plot_h - lh;
        let _ = write!(
            bars,
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" class="ref"/>
<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" class="live"/>"#,
            x0 + slot * 0.10,
            ry,
            bar_w,
            rh,
            x0 + slot * 0.52,
            ly,
            bar_w,
            lh,
        );
    }
    let baseline = pad_t + plot_h;
    format!(
        r#"<svg viewBox="0 0 {w:.0} {h:.0}" class="hist" preserveAspectRatio="none">
<line x1="{pad_l:.1}" y1="{baseline:.1}" x2="{:.1}" y2="{baseline:.1}" class="axis"/>{bars}</svg>"#,
        w - pad_l
    )
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root {
    --bg:#0f1115; --panel:#181b22; --border:#272b34; --fg:#e6e9ef; --muted:#9aa3b2;
    --ok:#2ea043; --ok-bg:#0f2a17; --bad:#f85149; --bad-bg:#2a1113;
    --ref:#58a6ff; --live:#f0883e; --axis:#3a4150;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg:#f6f8fa; --panel:#fff; --border:#d0d7de; --fg:#1f2328; --muted:#656d76;
      --ok:#1a7f37; --ok-bg:#dafbe1; --bad:#cf222e; --bad-bg:#ffebe9;
      --ref:#0969da; --live:#bc4c00; --axis:#afb8c1;
    }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:820px; margin:0 auto; padding:28px 16px 64px; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:15px; margin:0 0 10px; }
  .muted { color:var(--muted); font-size:13px; }
  .banner { margin:18px 0; padding:14px 18px; border-radius:10px; font-weight:600;
    border:1px solid var(--border); display:flex; align-items:center; gap:10px; }
  .banner.ok { background:var(--ok-bg); border-color:var(--ok); }
  .banner.bad { background:var(--bad-bg); border-color:var(--bad); }
  .dot { width:11px; height:11px; border-radius:50%; flex:none; }
  .dot.ok { background:var(--ok); } .dot.bad { background:var(--bad); }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:16px 18px; margin-top:16px; }
  .pill { display:inline-block; padding:1px 9px; border-radius:999px; font-size:12px;
    font-weight:600; vertical-align:middle; margin-left:6px; }
  .pill.ok { color:var(--ok); background:var(--ok-bg); }
  .pill.bad { color:var(--bad); background:var(--bad-bg); }
  .fhead { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; }
  .fbody { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; margin-top:8px; }
  .chart { flex:1 1 320px; min-width:280px; }
  svg.hist { width:100%; height:150px; display:block; }
  .hist .ref { fill:var(--ref); } .hist .live { fill:var(--live); }
  .hist .axis { stroke:var(--axis); stroke-width:1; }
  .metrics { list-style:none; padding:0; margin:0; font-variant-numeric:tabular-nums; }
  .metrics li { padding:2px 0; }
  .legend { color:var(--muted); font-size:12px; margin-top:6px; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; }
  .sw.ref { background:var(--ref); } .sw.live { background:var(--live); }
  .sumrow { display:flex; align-items:center; gap:12px; margin:6px 0; }
  .sumname { flex:0 0 130px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .sumbar { position:relative; flex:1; height:14px; background:var(--bg);
    border:1px solid var(--border); border-radius:4px; }
  .sumfill { height:100%; border-radius:3px; }
  .sumfill.ok { background:var(--ok); } .sumfill.bad { background:var(--bad); }
  .sumtick { position:absolute; top:-3px; width:2px; height:20px; background:var(--fg); opacity:.6; }
  .sumval { flex:0 0 130px; text-align:right; font-variant-numeric:tabular-nums; font-size:13px; }
  .tickmark { display:inline-block; width:2px; height:12px; background:var(--fg); vertical-align:middle; opacity:.6; }
  footer { margin-top:28px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>{{TITLE}}</h1>
  <div class="muted">generated by driftwatch</div>
  {{BODY}}
  <footer>Reference vs live distributions · powered by driftwatch</footer>
</div>
</body>
</html>"#;
