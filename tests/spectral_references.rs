//! Exact and statistical RMT references for rmt's spectral quantities.
//!
//! Deterministic exact checks (MP support closed form, density normalization)
//! plus seeded statistical checks (a sampled GOE has mean level-spacing ratio
//! ≈0.535 per Atas et al.; a sampled Wishart W/n sits inside the MP support).
//! rmt ships no eigensolver, so these hand-roll a cyclic Jacobi solver.

#![allow(clippy::needless_range_loop)] // Jacobi rotations index by (p,q,k) inherently

use ndarray::Array2;
use rand::{rngs::StdRng, SeedableRng};
use rmt::{
    marchenko_pastur_density, marchenko_pastur_support, mean_spacing_ratio, sample_goe_with,
    sample_wishart_with, wigner_semicircle_density,
};

/// Eigenvalues of a symmetric matrix via cyclic Jacobi rotations (ascending).
fn jacobi_eigenvalues(mat: &Array2<f64>) -> Vec<f64> {
    let n = mat.nrows();
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| mat[[i, j]]).collect())
        .collect();
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-9 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ev
}

fn integrate(a: f64, b: f64, steps: usize, f: impl Fn(f64) -> f64) -> f64 {
    let h = (b - a) / steps as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..steps {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

#[test]
fn marchenko_pastur_support_closed_form() {
    // σ²(1±√γ)², γ = min(ratio, 1/ratio); hand-computed known cases.
    for (ratio, sigma_sq, lo, hi) in [
        (1.0, 1.0, 0.0, 4.0),
        (0.25, 1.0, 0.25, 2.25),
        (0.25, 2.0, 0.5, 4.5),
        (4.0, 1.0, 0.25, 2.25), // symmetrized: min(4, 0.25) = 0.25
    ] {
        let (l, h) = marchenko_pastur_support(ratio, sigma_sq);
        assert!(
            (l - lo).abs() < 1e-9 && (h - hi).abs() < 1e-9,
            "MP support(c={ratio}, σ²={sigma_sq}) = ({l:.4},{h:.4}), want ({lo},{hi})"
        );
    }
}

#[test]
fn marchenko_pastur_density_integrates_to_one() {
    let (lo, hi) = marchenko_pastur_support(0.25, 1.0);
    let mass = integrate(lo, hi, 50000, |x| marchenko_pastur_density(x, 0.25, 1.0));
    assert!((mass - 1.0).abs() < 1e-2, "MP density mass {mass:.4} != 1");
}

#[test]
fn wigner_semicircle_normalized_and_peak() {
    let mass = integrate(-2.0, 2.0, 50000, |x| wigner_semicircle_density(x, 1.0));
    assert!((mass - 1.0).abs() < 1e-2, "semicircle mass {mass:.4} != 1");
    let peak = wigner_semicircle_density(0.0, 1.0);
    assert!(
        (peak - 1.0 / std::f64::consts::PI).abs() < 1e-2,
        "semicircle(0) {peak:.4} != 1/π"
    );
}

#[test]
fn goe_mean_spacing_ratio_matches_atas() {
    // GOE bulk level-spacing ratio ≈ 0.5359 (Atas et al. 2013); Poisson ≈ 0.386.
    let mut rng = StdRng::seed_from_u64(0xA7A5);
    let goe = sample_goe_with(&mut rng, 300);
    let ev = jacobi_eigenvalues(&goe);
    let r = mean_spacing_ratio(&ev);
    assert!(
        (0.50..=0.57).contains(&r),
        "GOE mean spacing ratio {r:.4} not ≈0.535 (Poisson would be ≈0.386)"
    );
}

#[test]
fn wishart_spectrum_within_mp_support() {
    let mut rng = StdRng::seed_from_u64(0x1234);
    let (n, p) = (600, 150);
    let w = sample_wishart_with(&mut rng, n, p);
    let ev: Vec<f64> = jacobi_eigenvalues(&w)
        .into_iter()
        .map(|v| v / n as f64)
        .collect();
    let (lo, hi) = marchenko_pastur_support(p as f64 / n as f64, 1.0);
    let inside = ev
        .iter()
        .filter(|&&v| v >= lo - 0.05 && v <= hi + 0.05)
        .count();
    let frac = inside as f64 / ev.len() as f64;
    assert!(
        frac > 0.95,
        "Wishart W/n spectrum {frac:.3} inside MP support, want >0.95"
    );
}
