//! # rmt
//!
//! Random Matrix Theory: eigenvalue distributions and spectral statistics.
//!
//! ## The Core Idea
//!
//! When you have a large random matrix, its eigenvalues follow predictable
//! distributions. This is surprising: randomness at the element level produces
//! order at the spectral level.
//!
//! ## Key Distributions
//!
//! | Distribution | Matrix Type | Density |
//! |--------------|-------------|---------|
//! | [`marchenko_pastur_density`] | Wishart (X^T X) | Bounded support |
//! | [`wigner_semicircle_density`] | Symmetric random | Semicircle |
//! | Tracy--Widom (edge; not implemented here) | Largest eigenvalue | Skewed |
//!
//! ## Quick Start
//!
//! ```rust
//! use rmt::{marchenko_pastur_density, wigner_semicircle_density, sample_wishart};
//! use ndarray::Array2;
//!
//! // Marchenko-Pastur: eigenvalue density of X^T X / n
//! let ratio = 0.5;  // p/n (dimensions / samples)
//! let density = marchenko_pastur_density(1.5, ratio, 1.0);
//!
//! // Wigner semicircle: eigenvalue density of symmetric matrix
//! let density = wigner_semicircle_density(0.5, 1.0);
//!
//! // Sample a Wishart matrix
//! let (n, p) = (100, 50);
//! let wishart = sample_wishart(n, p);
//! ```
//!
//! ## Why RMT for ML?
//!
//! - **Covariance matrices**: Sample covariance eigenvalues follow Marchenko-Pastur
//! - **Neural networks**: Weight matrix spectra reveal training dynamics
//! - **PCA**: Distinguish signal from noise eigenvalues
//! - **Regularization**: Set shrinkage based on spectral distribution
//!
//! ## The Marchenko-Pastur Law
//!
//! For a matrix X (n samples, p features), the eigenvalues of X^T X / n
//! cluster in [lambda_-, lambda_+] where:
//!
//! ```text
//! lambda_+/- = sigma^2 (1 +/- sqrt(p/n))^2
//!
//! Density: rho(lambda) = (1/(2*pi*sigma^2)) * sqrt((lambda_+ - lambda)(lambda - lambda_-)) / (gamma*lambda)
//! ```
//!
//! When p/n -> 0, this converges to a point mass at sigma^2 (classical regime).
//! When p/n > 0, eigenvalues spread (high-dimensional regime).
//!
//! ## The Wigner Semicircle
//!
//! For a symmetric matrix with i.i.d. entries, eigenvalues follow:
//!
//! ```text
//! rho(lambda) = (1/(2*pi*sigma^2)) * sqrt(4*sigma^2 - lambda^2)  for |lambda| <= 2*sigma
//! ```
//!
//! ## What Can Go Wrong
//!
//! 1. **Finite size effects**: MP/semicircle are asymptotic. Small n deviates.
//! 2. **Not centered**: MP assumes zero-mean data. Center your features.
//! 3. **Correlated features**: MP assumes independence. Correlated data has different spectrum.
//! 4. **Ratio out of range**: MP needs p/n in (0, infinity). Tracy-Widom for edge.
//! 5. **Numerical eigendecomposition**: For large matrices, use iterative methods.
//!
//! ## References
//!
//! - Marchenko & Pastur (1967). "Distribution of eigenvalues for some sets of random matrices"
//! - Wigner (1955). "Characteristic vectors of bordered matrices with infinite dimensions"
//! - Johnstone (2001). "On the distribution of the largest eigenvalue in PCA"

use std::f64::consts::PI;

use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Marchenko-Pastur density at point lambda.
///
/// For the eigenvalues of (1/n) X^T X where X is n x p with i.i.d. N(0, sigma^2) entries.
///
/// # Arguments
///
/// * `lambda` - Eigenvalue to evaluate density at
/// * `ratio` - gamma = p/n ratio (any positive value; values > 1 are folded via min(gamma, 1/gamma))
/// * `sigma_sq` - Variance of matrix entries (default 1.0)
///
/// # Returns
///
/// Density rho(lambda), or 0 if outside support [lambda_-, lambda_+]
///
/// # Example
///
/// ```rust
/// use rmt::marchenko_pastur_density;
///
/// let gamma = 0.5;  // p/n = 0.5
/// let density = marchenko_pastur_density(1.5, gamma, 1.0);
/// assert!(density > 0.0);
/// ```
pub fn marchenko_pastur_density(lambda: f64, ratio: f64, sigma_sq: f64) -> f64 {
    if ratio <= 0.0 || lambda <= 0.0 {
        return 0.0;
    }

    let gamma = ratio.min(1.0 / ratio); // Handle both p/n < 1 and p/n > 1
    let lambda_plus = sigma_sq * (1.0 + gamma.sqrt()).powi(2);
    let lambda_minus = sigma_sq * (1.0 - gamma.sqrt()).powi(2);

    if lambda < lambda_minus || lambda > lambda_plus {
        return 0.0;
    }

    let sqrt_term = ((lambda_plus - lambda) * (lambda - lambda_minus)).sqrt();
    sqrt_term / (2.0 * PI * sigma_sq * gamma * lambda)
}

/// Marchenko-Pastur support bounds [lambda_-, lambda_+].
///
/// # Arguments
///
/// * `ratio` - gamma = p/n ratio (any positive value; values > 1 are folded via min(gamma, 1/gamma))
/// * `sigma_sq` - Variance of matrix entries
///
/// # Returns
///
/// (lambda_minus, lambda_plus)
pub fn marchenko_pastur_support(ratio: f64, sigma_sq: f64) -> (f64, f64) {
    let gamma = ratio.min(1.0 / ratio);
    let lambda_plus = sigma_sq * (1.0 + gamma.sqrt()).powi(2);
    let lambda_minus = sigma_sq * (1.0 - gamma.sqrt()).powi(2);
    (lambda_minus, lambda_plus)
}

/// Wigner semicircle density at point lambda.
///
/// For eigenvalues of symmetric matrix with i.i.d. entries of variance sigma^2.
///
/// # Arguments
///
/// * `lambda` - Eigenvalue to evaluate density at
/// * `sigma` - Standard deviation (radius = 2*sigma)
///
/// # Returns
///
/// Density rho(lambda), or 0 if |lambda| > 2*sigma
///
/// # Example
///
/// ```rust
/// use rmt::wigner_semicircle_density;
///
/// // At lambda=0 with sigma=1, R=2, density = 2/(pi*R^2) * R = 1/pi
/// let density = wigner_semicircle_density(0.0, 1.0);
/// assert!(density > 0.3);  // Should be ~1/pi ~ 0.318
/// ```
pub fn wigner_semicircle_density(lambda: f64, sigma: f64) -> f64 {
    let r = 2.0 * sigma;
    if lambda.abs() > r {
        return 0.0;
    }

    (2.0 / (PI * r * r)) * (r * r - lambda * lambda).sqrt()
}

/// Sample a Wishart matrix W = X^T X where X is n x p Gaussian, using the
/// provided RNG for reproducibility.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `n` - Number of samples (rows of X)
/// * `p` - Number of features (columns of X)
///
/// # Returns
///
/// p x p Wishart matrix
pub fn sample_wishart_with<R: Rng>(rng: &mut R, n: usize, p: usize) -> Array2<f64> {
    let normal = Normal::new(0.0, 1.0).expect("Normal(0, 1) should be valid");
    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = normal.sample(rng);
        }
    }
    x.t().dot(&x)
}

/// Sample a Wishart matrix: W = X^T X where X is n x p Gaussian.
///
/// The eigenvalues of W/n follow the Marchenko-Pastur distribution
/// as n, p -> infinity with p/n -> gamma.
///
/// # Arguments
///
/// * `n` - Number of samples (rows of X)
/// * `p` - Number of features (columns of X)
///
/// # Returns
///
/// p x p Wishart matrix
pub fn sample_wishart(n: usize, p: usize) -> Array2<f64> {
    sample_wishart_with(&mut rand::rng(), n, p)
}

/// Sample a GOE (Gaussian Orthogonal Ensemble) matrix using the provided RNG
/// for reproducibility.
///
/// Symmetric matrix with Gaussian entries. Eigenvalues follow Wigner semicircle.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `n` - Matrix dimension
///
/// # Returns
///
/// n x n symmetric random matrix
pub fn sample_goe_with<R: Rng>(rng: &mut R, n: usize) -> Array2<f64> {
    let normal = Normal::new(0.0, 1.0).expect("Normal(0, 1) should be valid");
    let mut m = Array2::zeros((n, n));
    for i in 0..n {
        m[[i, i]] = normal.sample(rng) * 2.0_f64.sqrt();
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let val = normal.sample(rng);
            m[[i, j]] = val;
            m[[j, i]] = val;
        }
    }
    let scale = 1.0 / (n as f64).sqrt();
    m *= scale;
    m
}

/// Sample a GOE (Gaussian Orthogonal Ensemble) matrix.
///
/// Symmetric matrix with Gaussian entries. Eigenvalues follow Wigner semicircle.
///
/// # Arguments
///
/// * `n` - Matrix dimension
///
/// # Returns
///
/// n x n symmetric random matrix
pub fn sample_goe(n: usize) -> Array2<f64> {
    sample_goe_with(&mut rand::rng(), n)
}

/// Level spacing ratio for eigenvalue sequence.
///
/// The ratio r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}) where s_i = lambda_{i+1} - lambda_i.
/// For GOE: mean ~ 0.5307. For Poisson (uncorrelated): mean ~ 0.3863.
///
/// # Arguments
///
/// * `eigenvalues` - Sorted eigenvalue sequence
///
/// # Returns
///
/// Vector of spacing ratios
pub fn level_spacing_ratios(eigenvalues: &[f64]) -> Vec<f64> {
    if eigenvalues.len() < 3 {
        return vec![];
    }

    let n = eigenvalues.len();
    let mut ratios = Vec::with_capacity(n - 2);

    for i in 0..(n - 2) {
        let s1 = eigenvalues[i + 1] - eigenvalues[i];
        let s2 = eigenvalues[i + 2] - eigenvalues[i + 1];

        if s1 > 0.0 && s2 > 0.0 {
            ratios.push(s1.min(s2) / s1.max(s2));
        }
    }

    ratios
}

/// Mean level spacing ratio.
///
/// GOE (correlated): ~0.5307
/// Poisson (uncorrelated): ~0.3863
pub fn mean_spacing_ratio(eigenvalues: &[f64]) -> f64 {
    let ratios = level_spacing_ratios(eigenvalues);
    if ratios.is_empty() {
        0.0
    } else {
        ratios.iter().sum::<f64>() / (ratios.len() as f64)
    }
}

/// Empirical spectral density via histogram.
///
/// # Arguments
///
/// * `eigenvalues` - Eigenvalue samples
/// * `bins` - Number of histogram bins
///
/// # Returns
///
/// (bin_centers, densities)
pub fn empirical_spectral_density(eigenvalues: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    if eigenvalues.is_empty() || bins == 0 {
        return (vec![], vec![]);
    }

    let min = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-10 {
        return (vec![min], vec![1.0]);
    }

    let bin_width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];

    for &ev in eigenvalues {
        let idx = ((ev - min) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    let n = eigenvalues.len() as f64;
    let centers: Vec<f64> = (0..bins)
        .map(|i| min + (i as f64 + 0.5) * bin_width)
        .collect();
    let densities: Vec<f64> = counts.iter().map(|&c| c as f64 / (n * bin_width)).collect();

    (centers, densities)
}

/// Stieltjes transform: m(z) = (1/n) sum 1/(lambda_i - z)
///
/// The Stieltjes transform encodes the spectral distribution and is
/// central to proving limiting theorems in RMT.
pub fn stieltjes_transform(eigenvalues: &[f64], z: f64) -> f64 {
    let n = eigenvalues.len() as f64;
    eigenvalues.iter().map(|&ev| 1.0 / (ev - z)).sum::<f64>() / n
}

/// Estimate effective dimensionality of an embedding matrix using the Marchenko-Pastur law.
///
/// Given eigenvalues of the sample covariance matrix (X^T X / n), counts how many
/// exceed the upper edge of the Marchenko-Pastur bulk (lambda_+). Eigenvalues above
/// this threshold are signal; those below are noise.
///
/// This is the standard RMT approach to PCA dimensionality selection: instead of
/// choosing an arbitrary variance-explained cutoff, use the theoretically-derived
/// noise boundary.
///
/// # Arguments
///
/// * `eigenvalues` - Eigenvalues of the sample covariance matrix, in any order
/// * `n_samples` - Number of samples used to compute the covariance
/// * `n_features` - Number of features (dimensionality of the data)
///
/// # Returns
///
/// Number of eigenvalues exceeding the MP upper edge (signal dimensions)
///
/// # Example
///
/// ```rust
/// use rmt::effective_dimension;
///
/// // 5 signal eigenvalues + 95 noise eigenvalues
/// let mut eigenvalues = vec![10.0, 8.0, 6.0, 4.0, 2.5];
/// eigenvalues.extend(vec![1.0; 95]); // noise floor
/// let dim = effective_dimension(&eigenvalues, 200, 100);
/// assert!(dim >= 4 && dim <= 6, "got {dim}");
/// ```
pub fn effective_dimension(eigenvalues: &[f64], n_samples: usize, n_features: usize) -> usize {
    if eigenvalues.is_empty() || n_samples == 0 || n_features == 0 {
        return 0;
    }

    let ratio = n_features as f64 / n_samples as f64;

    // Estimate noise variance as the median eigenvalue (robust to outliers).
    // For pure noise, eigenvalues cluster around sigma^2.
    let mut sorted: Vec<f64> = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // sigma^2 estimate: under MP, the median of noise eigenvalues ≈ sigma^2
    // for moderate ratios. Use a slightly more conservative estimate.
    let sigma_sq = median.max(1e-12);

    let (_, lambda_plus) = marchenko_pastur_support(ratio, sigma_sq);

    eigenvalues.iter().filter(|&&ev| ev > lambda_plus).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marchenko_pastur_normalization() {
        // Density should integrate to ~1
        let ratio = 0.5;
        let (lambda_min, lambda_max) = marchenko_pastur_support(ratio, 1.0);

        let n_points = 1000;
        let dx = (lambda_max - lambda_min) / n_points as f64;
        let integral: f64 = (0..n_points)
            .map(|i| {
                let x = lambda_min + (i as f64 + 0.5) * dx;
                marchenko_pastur_density(x, ratio, 1.0) * dx
            })
            .sum();

        assert!(
            (integral - 1.0).abs() < 0.05,
            "MP density should integrate to ~1"
        );
    }

    #[test]
    fn test_wigner_semicircle_normalization() {
        let sigma = 1.0;
        let r = 2.0 * sigma;

        let n_points = 1000;
        let dx = 2.0 * r / n_points as f64;
        let integral: f64 = (0..n_points)
            .map(|i| {
                let x = -r + (i as f64 + 0.5) * dx;
                wigner_semicircle_density(x, sigma) * dx
            })
            .sum();

        assert!(
            (integral - 1.0).abs() < 0.05,
            "Wigner density should integrate to ~1"
        );
    }

    #[test]
    fn test_wigner_at_zero() {
        let density = wigner_semicircle_density(0.0, 1.0);
        let expected = 1.0 / PI;
        assert!(
            (density - expected).abs() < 0.01,
            "Wigner at zero: {} vs expected {}",
            density,
            expected
        );
    }

    #[test]
    fn test_wishart_shape() {
        let wishart = sample_wishart(100, 50);
        assert_eq!(wishart.shape(), &[50, 50]);
    }

    #[test]
    fn test_wishart_seeded_deterministic() {
        use rand::SeedableRng;
        let mut rng1 = rand::rngs::SmallRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::SmallRng::seed_from_u64(42);
        let w1 = sample_wishart_with(&mut rng1, 20, 10);
        let w2 = sample_wishart_with(&mut rng2, 20, 10);
        assert_eq!(w1, w2);
    }

    #[test]
    fn test_goe_symmetric() {
        let goe = sample_goe(10);
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (goe[[i, j]] - goe[[j, i]]).abs() < 1e-10,
                    "GOE should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_goe_seeded_deterministic() {
        use rand::SeedableRng;
        let mut rng1 = rand::rngs::SmallRng::seed_from_u64(99);
        let mut rng2 = rand::rngs::SmallRng::seed_from_u64(99);
        let g1 = sample_goe_with(&mut rng1, 15);
        let g2 = sample_goe_with(&mut rng2, 15);
        assert_eq!(g1, g2);
    }

    #[test]
    fn test_effective_dimension_with_signal() {
        // Create eigenvalues with clear signal: 5 large + 95 noise
        let mut eigenvalues = vec![10.0, 8.0, 6.0, 4.0, 3.0];
        eigenvalues.extend(vec![1.0; 95]);
        let dim = effective_dimension(&eigenvalues, 200, 100);
        assert!(dim >= 4 && dim <= 6, "expected 4-6 signal dims, got {dim}");
    }

    #[test]
    fn test_effective_dimension_pure_noise() {
        // All eigenvalues are noise (clustered around 1.0)
        let eigenvalues = vec![1.0; 50];
        let dim = effective_dimension(&eigenvalues, 200, 50);
        assert_eq!(dim, 0, "pure noise should have 0 effective dims");
    }

    #[test]
    fn test_effective_dimension_empty() {
        assert_eq!(effective_dimension(&[], 100, 50), 0);
    }

    #[test]
    fn test_spacing_ratio_bounds() {
        let eigenvalues = vec![1.0, 2.0, 3.5, 4.0, 6.0];
        let ratios = level_spacing_ratios(&eigenvalues);

        for r in ratios {
            assert!(
                (0.0..=1.0).contains(&r),
                "spacing ratio should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_empirical_density() {
        let eigenvalues: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let (centers, densities) = empirical_spectral_density(&eigenvalues, 10);

        assert_eq!(centers.len(), 10);
        assert_eq!(densities.len(), 10);

        // All densities should be roughly equal for uniform input
        for d in &densities {
            assert!(*d > 0.5 && *d < 1.5);
        }
    }
}
