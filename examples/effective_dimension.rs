//! Signal-vs-noise dimension selection via the Marchenko-Pastur edge.
//!
//! Builds a synthetic covariance spectrum: a few large "signal" eigenvalues
//! planted above a noise bulk near sigma^2 = 1. The Marchenko-Pastur upper
//! edge `lambda_+` is the theoretical ceiling of the noise bulk; eigenvalues
//! above it are signal. This is the RMT alternative to an arbitrary
//! variance-explained cutoff in PCA.
//!
//! Run with: `cargo run --example effective_dimension`

use rmt::{effective_dimension, empirical_spectral_density, marchenko_pastur_support};

fn main() {
    let (n_samples, n_features) = (400usize, 100usize);
    let gamma = n_features as f64 / n_samples as f64;

    // Synthetic spectrum of X^T X / n: 4 signal eigenvalues above a unit-scale
    // noise bulk. With sigma^2 = 1 these are the eigenvalues you would see from
    // a covariance matrix whose data has 4 real directions plus isotropic noise.
    let mut eigenvalues = vec![14.0, 9.0, 5.5, 3.0];
    eigenvalues.extend(vec![1.0; n_features - eigenvalues.len()]);

    let (lambda_minus, lambda_plus) = marchenko_pastur_support(gamma, 1.0);
    println!("p/n ratio (gamma):        {gamma:.3}");
    println!("MP noise support:         [{lambda_minus:.3}, {lambda_plus:.3}]");

    let signal_dims = effective_dimension(&eigenvalues, n_samples, n_features);
    println!("eigenvalues above edge:   {signal_dims} (planted 4)");

    // Coarse histogram of the spectrum.
    let (centers, density) = empirical_spectral_density(&eigenvalues, 8);
    println!("empirical spectral density (center: density):");
    for (c, d) in centers.iter().zip(&density) {
        println!("  {c:6.2}: {d:.4}");
    }
}
