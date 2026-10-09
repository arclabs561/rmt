//! `effective_dimension` sorts the spectrum to find its median. A NaN
//! eigenvalue must not panic the sort.

use rmt::effective_dimension;

#[test]
fn effective_dimension_with_nan_eigenvalue_does_not_panic() {
    let mut eigenvalues: Vec<f64> = (0..40).map(|i| 0.5 + i as f64 * 0.02).collect();
    eigenvalues[7] = f64::NAN;
    eigenvalues[23] = f64::NAN;
    let _ = effective_dimension(&eigenvalues, 200, 40);
}
