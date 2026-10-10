# Changelog

## [Unreleased]

## [0.1.4] - 2026-10-09

### Changed

- Declare `rust-version = "1.75"`.

### Fixed

- `marchenko_pastur_support` and `marchenko_pastur_density` no longer fold
  `gamma > 1` to `1 / gamma`. For `gamma = 4` the support is now `(1, 9)`,
  the range of the nonzero eigenvalues of `X^T X / n`, instead of
  `(0.25, 2.25)`. For `gamma > 1` the density omits the atom of mass
  `1 - 1/gamma` at 0 and integrates to `1/gamma` over the support.
- `effective_dimension` estimates the noise level as the median eigenvalue
  divided by the Marchenko-Pastur median (Gavish & Donoho 2014) instead of
  the raw median, and leaves the `p - n` zero eigenvalues out of the median
  when `p > n`. Pure noise previously reported about 10% of the dimensions as
  signal; results differ from 0.1.3 for the same input.
- `effective_dimension` sorts with `total_cmp`; the previous comparator
  could panic on NaN eigenvalues under Rust 1.81 and later.

## [0.1.3] - 2026-06-10

### Added

- Property tests for Marchenko-Pastur density, Wigner surmise, spacing ratios, and effective dimension.
- PCA dimensionality selection example for `effective_dimension`.

