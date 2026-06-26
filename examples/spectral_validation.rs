//! Validate rmt against exact random-matrix-theory references.
//!
//! Mixes exact checks and famous statistical constants:
//! - Marchenko-Pastur support equals the closed form σ²(1±√γ)² (hand-computed
//!   known cases);
//! - the MP density and the Wigner semicircle each integrate to 1 over support;
//! - a sampled Wishart W/n has its spectrum inside the MP support;
//! - a sampled GOE has mean level-spacing ratio ≈ 0.535 (Atas et al. 2013),
//!   versus ≈0.386 for an uncorrelated (Poisson) spectrum.
//!
//! rmt analyzes eigenvalue lists but ships no eigensolver, so this example
//! hand-rolls a cyclic Jacobi solver for symmetric matrices (self-contained).
//!
//! ```sh
//! cargo run --release --example spectral_validation
//! ```

use std::process::ExitCode;

use ndarray::Array2;
use rmt::{
    marchenko_pastur_density, marchenko_pastur_support, mean_spacing_ratio, sample_goe,
    sample_wishart, wigner_semicircle_density,
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

/// Trapezoid integral of `f` over [a, b] with `steps` points.
fn integrate(a: f64, b: f64, steps: usize, f: impl Fn(f64) -> f64) -> f64 {
    let h = (b - a) / steps as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..steps {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

fn main() -> ExitCode {
    let mut failures = 0u64;
    let mut check = |cond: bool, what: String| {
        if !cond {
            failures += 1;
            eprintln!("  VIOLATION: {what}");
        }
    };

    // 1. MP support == closed form (hand-computed known cases).
    let cases = [
        (1.0, 1.0, 0.0, 4.0),
        (0.25, 1.0, 0.25, 2.25),
        (0.25, 2.0, 0.5, 4.5),
    ];
    for (ratio, sigma_sq, lo, hi) in cases {
        let (l, h) = marchenko_pastur_support(ratio, sigma_sq);
        check(
            (l - lo).abs() < 1e-9 && (h - hi).abs() < 1e-9,
            format!("MP support(c={ratio}, σ²={sigma_sq}) = ({l:.4},{h:.4}), want ({lo},{hi})"),
        );
    }

    // 2. MP density integrates to ~1 over its support.
    let (lo, hi) = marchenko_pastur_support(0.25, 1.0);
    let mp_mass = integrate(lo, hi, 20000, |x| marchenko_pastur_density(x, 0.25, 1.0));
    check(
        (mp_mass - 1.0).abs() < 1e-2,
        format!("MP density mass = {mp_mass:.4}, want ~1"),
    );

    // 3. Wigner semicircle: integrates to ~1 over [-2σ, 2σ]; value at 0 ≈ 1/π.
    let sc_mass = integrate(-2.0, 2.0, 20000, |x| wigner_semicircle_density(x, 1.0));
    check(
        (sc_mass - 1.0).abs() < 1e-2,
        format!("semicircle mass = {sc_mass:.4}, want ~1"),
    );
    let sc0 = wigner_semicircle_density(0.0, 1.0);
    check(
        (sc0 - 1.0 / std::f64::consts::PI).abs() < 1e-2,
        format!("semicircle(0) = {sc0:.4}, want ~1/π"),
    );

    // 4. GOE mean spacing ratio ≈ 0.535 (Atas et al.), well above Poisson 0.386.
    let goe = sample_goe(250);
    let goe_ev = jacobi_eigenvalues(&goe);
    let r = mean_spacing_ratio(&goe_ev);
    check(
        (0.50..=0.57).contains(&r),
        format!("GOE mean spacing ratio = {r:.4}, want ~0.535 (Poisson would be ~0.386)"),
    );
    println!("GOE(250) mean spacing ratio = {r:.4} (literature 0.5359)");

    // 5. Wishart W/n spectrum lies inside the MP support.
    let (n, p) = (600, 150);
    let w = sample_wishart(n, p);
    let mut wev = jacobi_eigenvalues(&w);
    for v in &mut wev {
        *v /= n as f64; // eigenvalues of W/n follow MP
    }
    let (mlo, mhi) = marchenko_pastur_support(p as f64 / n as f64, 1.0);
    let inside = wev
        .iter()
        .filter(|&&v| v >= mlo - 0.05 && v <= mhi + 0.05)
        .count();
    let frac = inside as f64 / wev.len() as f64;
    check(
        frac > 0.95,
        format!("Wishart spectrum in MP support: {frac:.3} (want >0.95)"),
    );
    println!(
        "Wishart W/n (c={:.3}) spectrum {:.1}% inside MP support [{mlo:.3},{mhi:.3}]",
        p as f64 / n as f64,
        frac * 100.0
    );

    if failures == 0 {
        println!("\nPASS: rmt matches exact RMT references");
        ExitCode::SUCCESS
    } else {
        eprintln!("\nFAIL: {failures} rmt reference violations");
        ExitCode::FAILURE
    }
}
