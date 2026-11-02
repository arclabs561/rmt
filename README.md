# rmt

[![crates.io](https://img.shields.io/crates/v/rmt.svg)](https://crates.io/crates/rmt)
[![Documentation](https://docs.rs/rmt/badge.svg)](https://docs.rs/rmt)

Random matrix theory primitives.

`rmt` works with eigenvalue spectra. It does not do eigendecomposition.

```toml
[dependencies]
rmt = "0.1.4"
```

```rust
use rmt::{marchenko_pastur_density, wigner_semicircle_density, sample_wishart};

// Marchenko-Pastur: eigenvalue density of sample covariance, gamma = p/n
let density = marchenko_pastur_density(1.5, 0.5, 1.0);

// Wigner semicircle: symmetric random matrix eigenvalues
let density = wigner_semicircle_density(0.5, 1.0);

// Sample a Wishart matrix W = X^T X, here 100 samples x 50 features
let wishart = sample_wishart(100, 50);
```

[`examples/effective_dimension.rs`](examples/effective_dimension.rs) plants
four signal eigenvalues in a synthetic spectrum and recovers them against the
Marchenko-Pastur edge:

```sh
cargo run --example effective_dimension
```

```text
p/n ratio (gamma):        0.250
MP noise support:         [0.250, 2.250]
eigenvalues above edge:   4 (planted 4)
```

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

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.
