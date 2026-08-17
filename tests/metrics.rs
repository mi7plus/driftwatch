//! Hand-verified fixtures for the drift metrics.
//!
//! There is no single universally-famous benchmark dataset for PSI the way
//! Longley serves regression diagnostics, so this crate establishes its own
//! verified fixture: the expected values below were computed by hand once (see
//! the comments) and are the ground truth these metrics are held to.

use approx::assert_abs_diff_eq;
use driftwatch::{
    chi_square_test, js_divergence, kl_divergence, ks_test, psi, BinDefinition, Histogram,
};

/// Two 2-bin histograms: reference freqs [0.7, 0.3], live freqs [0.5, 0.5].
fn fixture_pair() -> (Histogram, Histogram) {
    let bins = BinDefinition::Continuous {
        edges: vec![0.0, 1.0, 2.0],
    };
    let reference = Histogram::new(bins.clone(), vec![70.0, 30.0]).unwrap();
    let live = Histogram::new(bins, vec![50.0, 50.0]).unwrap();
    (reference, live)
}

#[test]
fn psi_matches_hand_computed() {
    // PSI = (0.5-0.7)*ln(0.5/0.7) + (0.5-0.3)*ln(0.5/0.3)
    //     = 0.0672945 + 0.1021652 = 0.1694597
    let (reference, live) = fixture_pair();
    assert_abs_diff_eq!(psi(&reference, &live).unwrap(), 0.169_459_7, epsilon = 1e-4);
}

#[test]
fn kl_matches_hand_computed() {
    // D_KL(live||ref) = 0.5*ln(0.5/0.7) + 0.5*ln(0.5/0.3) = 0.0871769
    let (reference, live) = fixture_pair();
    assert_abs_diff_eq!(
        kl_divergence(&reference, &live).unwrap(),
        0.087_176_9,
        epsilon = 1e-4
    );
}

#[test]
fn js_matches_hand_computed_and_is_bounded() {
    // M = [0.6, 0.4]; JS = 0.5*KL(live||M) + 0.5*KL(ref||M) = 0.0210060
    let (reference, live) = fixture_pair();
    let js = js_divergence(&reference, &live).unwrap();
    assert_abs_diff_eq!(js, 0.021_006_0, epsilon = 1e-4);
    // JS is bounded to [0, ln 2].
    assert!((0.0..=std::f64::consts::LN_2).contains(&js));
}

#[test]
fn identical_histograms_have_zero_divergence() {
    let bins = BinDefinition::Continuous {
        edges: vec![0.0, 1.0, 2.0],
    };
    let a = Histogram::new(bins.clone(), vec![40.0, 60.0]).unwrap();
    let b = Histogram::new(bins, vec![40.0, 60.0]).unwrap();
    assert_abs_diff_eq!(psi(&a, &b).unwrap(), 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(kl_divergence(&a, &b).unwrap(), 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(js_divergence(&a, &b).unwrap(), 0.0, epsilon = 1e-6);
}

#[test]
fn zero_frequency_bin_does_not_produce_inf_or_nan() {
    // Reference has an empty second bin; without epsilon smoothing PSI/KL would
    // be inf/NaN. Smoothing must keep them finite.
    let bins = BinDefinition::Continuous {
        edges: vec![0.0, 1.0, 2.0],
    };
    let reference = Histogram::new(bins.clone(), vec![100.0, 0.0]).unwrap();
    let live = Histogram::new(bins, vec![50.0, 50.0]).unwrap();

    let p = psi(&reference, &live).unwrap();
    let kl = kl_divergence(&reference, &live).unwrap();
    let js = js_divergence(&reference, &live).unwrap();
    assert!(p.is_finite(), "PSI was {p}");
    assert!(kl.is_finite(), "KL was {kl}");
    assert!(js.is_finite(), "JS was {js}");
}

#[test]
fn bin_count_mismatch_errors() {
    let a = Histogram::new(
        BinDefinition::Continuous {
            edges: vec![0.0, 1.0, 2.0],
        },
        vec![1.0, 1.0],
    )
    .unwrap();
    let b = Histogram::new(
        BinDefinition::Continuous {
            edges: vec![0.0, 1.0],
        },
        vec![1.0],
    )
    .unwrap();
    assert!(psi(&a, &b).is_err());
}

#[test]
fn ks_identical_samples_zero_statistic() {
    let reference = [1.0, 2.0, 3.0, 4.0, 5.0];
    let live = [1.0, 2.0, 3.0, 4.0, 5.0];
    let r = ks_test(&reference, &live).unwrap();
    assert_eq!(r.statistic, 0.0);
    assert!(r.p_value > 0.99);
}

#[test]
fn ks_disjoint_samples_reject_null() {
    // Every reference value is below every live value: the ECDFs never overlap,
    // so D = 1.0 exactly and the null is rejected.
    let reference = [0.0, 1.0, 2.0, 3.0];
    let live = [4.0, 5.0, 6.0, 7.0];
    let r = ks_test(&reference, &live).unwrap();
    assert_abs_diff_eq!(r.statistic, 1.0, epsilon = 1e-12);
    assert!(r.p_value < 0.05, "p was {}", r.p_value);
}

#[test]
fn chi_square_matches_hand_computed() {
    // 2x2 table: ref [40, 60], live [60, 40]. All expected cells = 50.
    // stat = 4 * (10^2 / 50) = 8.0; df = 1; p = 1 - chi2cdf(8, 1) ≈ 0.004678.
    let bins = BinDefinition::Categorical {
        categories: vec!["a".into(), "b".into()],
    };
    let reference = Histogram::new(bins.clone(), vec![40.0, 60.0]).unwrap();
    let live = Histogram::new(bins, vec![60.0, 40.0]).unwrap();
    let r = chi_square_test(&reference, &live).unwrap();
    assert_abs_diff_eq!(r.statistic, 8.0, epsilon = 1e-9);
    assert_eq!(r.degrees_of_freedom, 1);
    assert_abs_diff_eq!(r.p_value, 0.004_678, epsilon = 1e-4);
    assert!(r.p_value < 0.05);
}

#[test]
fn chi_square_identical_distributions_not_significant() {
    let bins = BinDefinition::Categorical {
        categories: vec!["a".into(), "b".into()],
    };
    let reference = Histogram::new(bins.clone(), vec![50.0, 50.0]).unwrap();
    let live = Histogram::new(bins, vec![50.0, 50.0]).unwrap();
    let r = chi_square_test(&reference, &live).unwrap();
    assert_abs_diff_eq!(r.statistic, 0.0, epsilon = 1e-12);
    assert!(r.p_value > 0.99);
}
