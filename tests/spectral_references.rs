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
    effective_dimension, marchenko_pastur_density, marchenko_pastur_support, mean_spacing_ratio,
    sample_goe_with, sample_wishart_with, wigner_semicircle_density,
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
    ev.sort_by(|x, y| x.total_cmp(y));
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
    // σ²(1±√γ)² with γ = p/n unfolded; hand-computed known cases. For γ > 1
    // these bound the nonzero eigenvalues of XᵀX/n (the rest are exactly 0).
    for (ratio, sigma_sq, lo, hi) in [
        (1.0, 1.0, 0.0, 4.0),
        (0.25, 1.0, 0.25, 2.25),
        (0.25, 2.0, 0.5, 4.5),
        (4.0, 1.0, 1.0, 9.0),
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

#[test]
fn marchenko_pastur_continuous_mass_is_one_over_gamma_above_one() {
    // For γ = p/n > 1, MP(γ) has an atom of mass 1 − 1/γ at 0; the density
    // is the continuous part and carries the remaining 1/γ.
    let gamma = 4.0;
    let (lo, hi) = marchenko_pastur_support(gamma, 1.0);
    let mass = integrate(lo, hi, 50000, |x| marchenko_pastur_density(x, gamma, 1.0));
    assert!(
        (mass - 1.0 / gamma).abs() < 1e-2,
        "MP(γ=4) continuous mass {mass:.4} != 1/γ"
    );
}

#[test]
fn wishart_wide_spectrum_within_unfolded_mp_support() {
    // p > n: the n nonzero eigenvalues of W/n fill [(1−√γ)², (1+√γ)²] with
    // γ = p/n (numpy check at p=800, n=200 gave [0.966, 8.80] for γ=4).
    let mut rng = StdRng::seed_from_u64(0x5678);
    let (n, p) = (60, 240);
    let w = sample_wishart_with(&mut rng, n, p);
    let ev = jacobi_eigenvalues(&w);
    let nonzero: Vec<f64> = ev[p - n..].iter().map(|v| v / n as f64).collect();
    let (lo, hi) = marchenko_pastur_support(p as f64 / n as f64, 1.0);
    let top = nonzero[n - 1];
    let bottom = nonzero[0];
    assert!(
        (top - hi).abs() < 0.15 * hi && (bottom - lo).abs() < 0.5 * lo,
        "wide Wishart nonzero spectrum [{bottom:.3}, {top:.3}] vs MP support [{lo:.3}, {hi:.3}]"
    );
}

/// Quantile of the MP(γ, σ²=1) continuous part at probability `q`, by
/// integrating the closed-form density under λ = a + (b−a)·sin²φ, which
/// removes the square-root edges and the 1/√λ pole at γ = 1.
fn mp_quantiles(gamma: f64, probs: &[f64]) -> Vec<f64> {
    let a = (1.0 - gamma.sqrt()).powi(2);
    let b = (1.0 + gamma.sqrt()).powi(2);
    let steps = 200_000;
    let h = std::f64::consts::FRAC_PI_2 / steps as f64;
    let mut cdf = Vec::with_capacity(steps + 1);
    let mut lam = Vec::with_capacity(steps + 1);
    let mut acc = 0.0;
    cdf.push(0.0);
    lam.push(a);
    for i in 0..steps {
        let phi = (i as f64 + 0.5) * h;
        let x = a + (b - a) * phi.sin().powi(2);
        // dλ/dφ = (b−a)·sin 2φ
        let rho = ((b - x) * (x - a)).sqrt() / (2.0 * std::f64::consts::PI * gamma * x);
        acc += rho * (b - a) * (2.0 * phi).sin() * h;
        cdf.push(acc);
        let phi1 = (i as f64 + 1.0) * h;
        lam.push(a + (b - a) * phi1.sin().powi(2));
    }
    let total = acc;
    probs
        .iter()
        .map(|&q| {
            let target = q * total;
            let j = cdf.partition_point(|&c| c < target).clamp(1, steps);
            let t = (target - cdf[j - 1]) / (cdf[j] - cdf[j - 1]);
            lam[j - 1] + t * (lam[j] - lam[j - 1])
        })
        .collect()
}

#[test]
fn effective_dimension_zero_on_exact_mp_noise() {
    // Eigenvalues placed at the MP(σ²) quantiles are pure noise: nothing lies
    // above λ₊, for any σ² (the estimator must be scale-equivariant).
    let p = 200;
    for gamma in [0.25, 0.5, 1.0, 2.0] {
        let n = (p as f64 / gamma).round() as usize;
        let m = p.min(n);
        let probs: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5) / m as f64).collect();
        for sigma_sq in [1.0, 3.0] {
            let mut ev: Vec<f64> = mp_quantiles(gamma, &probs)
                .into_iter()
                .map(|v| v * sigma_sq)
                .collect();
            ev.extend(std::iter::repeat_n(0.0, p - m));
            let dim = effective_dimension(&ev, n, p);
            assert_eq!(dim, 0, "pure MP noise γ={gamma} σ²={sigma_sq}: {dim} dims");
        }
    }
}

#[test]
fn effective_dimension_recovers_spikes_over_mp_noise() {
    // Five spikes well above λ₊ on top of square MP noise are all found.
    let p = 200;
    let probs: Vec<f64> = (0..p - 5)
        .map(|i| (i as f64 + 0.5) / (p - 5) as f64)
        .collect();
    let mut ev = mp_quantiles(1.0, &probs);
    ev.extend([8.0, 9.0, 10.0, 12.0, 15.0]);
    assert_eq!(effective_dimension(&ev, p, p), 5);
}

#[test]
fn effective_dimension_small_on_sampled_noise() {
    // Sampled square Wishart noise: the largest eigenvalue crosses λ₊ only by
    // Tracy-Widom fluctuations, so at most a couple of dims, not ~10% of p.
    let mut rng = StdRng::seed_from_u64(0x9ABC);
    let (n, p) = (120, 120);
    let w = sample_wishart_with(&mut rng, n, p);
    let ev: Vec<f64> = jacobi_eigenvalues(&w)
        .into_iter()
        .map(|v| v / n as f64)
        .collect();
    let dim = effective_dimension(&ev, n, p);
    assert!(dim <= 2, "square pure-noise Wishart gave {dim} signal dims");
}
