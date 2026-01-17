# rmt

Random Matrix Theory primitives for spectral analysis and signal detection.
Implements Marchenko-Pastur law, Wigner semicircle law, and eigenvalue spacing statistics.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/rmt) | [docs.rs](https://docs.rs/rmt)

```rust
use rmt::{marchenko_pastur_density, wigner_semicircle_density, sample_wishart};

// Marchenko-Pastur: eigenvalue density of sample covariance
let ratio = 0.5;  // p/n
let density = marchenko_pastur_density(1.5, ratio, 1.0);

// Wigner semicircle: symmetric random matrix eigenvalues
let density = wigner_semicircle_density(0.5, 1.0);

// Sample a Wishart matrix
let wishart = sample_wishart(100, 50);
```

## Functions

| Function | Purpose |
|----------|---------|
| `marchenko_pastur_density` | MP law density |
| `marchenko_pastur_support` | MP support bounds |
| `wigner_semicircle_density` | Wigner law density |
| `sample_wishart` | Sample X^T X |
| `sample_goe` | Gaussian Orthogonal Ensemble |
| `level_spacing_ratios` | Eigenvalue spacing statistics |
| `empirical_spectral_density` | Histogram-based density |
| `stieltjes_transform` | m(z) transform |

## Why RMT?

- Covariance matrix eigenvalues follow MP distribution
- Neural network weight spectra reveal training dynamics
- Distinguish signal from noise in PCA
