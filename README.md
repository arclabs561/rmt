# rmt

[![crates.io](https://img.shields.io/crates/v/rmt.svg)](https://crates.io/crates/rmt)
[![Documentation](https://docs.rs/rmt/badge.svg)](https://docs.rs/rmt)
[![CI](https://github.com/arclabs561/rmt/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rmt/actions/workflows/ci.yml)

Random matrix theory primitives.

Closed-form eigenvalue laws (Marchenko-Pastur for sample covariance, Wigner
semicircle for symmetric matrices) plus diagnostics that run on an eigenvalue
spectrum: signal-vs-noise dimension counting against the Marchenko-Pastur edge,
level-spacing ratios, empirical spectral histograms, and the Stieltjes
transform. Samplers for Wishart and Gaussian Orthogonal Ensemble matrices are
included for generating test spectra. The diagnostics take eigenvalues you have
already computed; this crate does not do eigendecomposition itself.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/rmt) | [docs.rs](https://docs.rs/rmt)

```rust
use rmt::{marchenko_pastur_density, wigner_semicircle_density, sample_wishart};

// Marchenko-Pastur: eigenvalue density of sample covariance, gamma = p/n
let density = marchenko_pastur_density(1.5, 0.5, 1.0);

// Wigner semicircle: symmetric random matrix eigenvalues
let density = wigner_semicircle_density(0.5, 1.0);

// Sample a Wishart matrix W = X^T X, here 100 samples x 50 features
let wishart = sample_wishart(100, 50);
```

See [`examples/effective_dimension.rs`](examples/effective_dimension.rs) for
recovering signal dimensions from a synthetic spectrum
(`cargo run --example effective_dimension`).

## Functions

| Function | Purpose |
|----------|---------|
| `marchenko_pastur_density` | MP law density at an eigenvalue |
| `marchenko_pastur_support` | MP support bounds `(lambda_-, lambda_+)` |
| `wigner_semicircle_density` | Wigner semicircle density at an eigenvalue |
| `effective_dimension` | Count eigenvalues above the MP edge (signal dims) |
| `level_spacing_ratios` | Per-gap eigenvalue spacing ratios |
| `mean_spacing_ratio` | Mean spacing ratio (GOE ~0.531, Poisson ~0.386) |
| `empirical_spectral_density` | Histogram of an eigenvalue sample |
| `stieltjes_transform` | `m(z) = (1/n) sum 1/(lambda_i - z)` |
| `sample_wishart`, `sample_wishart_with` | Sample `X^T X` (latter takes an RNG) |
| `sample_goe`, `sample_goe_with` | Sample a GOE matrix (latter takes an RNG) |

## Why RMT?

- Covariance matrix eigenvalues follow the MP distribution
- Neural network weight spectra reveal training dynamics
- Distinguishing signal from noise eigenvalues in PCA
