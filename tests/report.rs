//! The static HTML report must be self-contained and reflect the drift result.

use driftwatch::{
    DatasetMonitor, EqualFrequencyBinning, HtmlReport, LiveFeature, ReferenceDistribution,
};

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

fn drifted_report() -> driftwatch::DriftReport {
    let mut monitor = DatasetMonitor::new();
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "latency",
            &linspace(0.0, 1.0, 500),
            EqualFrequencyBinning::new(10).unwrap(),
        )
        .unwrap(),
    );
    monitor.add_feature(
        ReferenceDistribution::fit_continuous(
            "amount",
            &linspace(0.0, 1.0, 500),
            EqualFrequencyBinning::new(10).unwrap(),
        )
        .unwrap(),
    );
    let shifted = linspace(2.0, 3.0, 500);
    let unchanged = linspace(0.0, 1.0, 500);
    monitor
        .check(&[
            ("latency", LiveFeature::Continuous(&shifted)),
            ("amount", LiveFeature::Continuous(&unchanged)),
        ])
        .unwrap()
}

#[test]
fn html_report_is_self_contained_and_reflects_drift() {
    let report = drifted_report();
    let html = HtmlReport::new(&report)
        .with_title("Nightly drift")
        .to_html();

    // Well-formed, titled document.
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains("</html>"));
    assert!(html.contains("Nightly drift"));

    // Reflects the per-feature verdicts.
    assert!(html.contains("latency"));
    assert!(html.contains("amount"));
    assert!(html.contains("DRIFTED"));

    // Draws charts and is self-contained (no external asset URLs).
    assert!(html.contains("<svg"));
    assert!(!html.contains("http://"));
    assert!(!html.contains("https://"));
}

#[test]
fn html_report_saves_to_disk() {
    let report = drifted_report();
    let path = std::env::temp_dir().join("driftwatch_test_report.html");
    HtmlReport::new(&report).save(&path).unwrap();
    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.contains("</html>"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn empty_report_still_renders() {
    let monitor = DatasetMonitor::new();
    let report = monitor.check(&[]).unwrap();
    let html = HtmlReport::new(&report).to_html();
    assert!(html.contains("</html>"));
}
