//! Binomial coefficients.

use crate::common::mul_div;

/// Binomial coefficient `C(n, k)`.
///
/// Returns `0` when `k > n`. Runs in `O(k)` time with `O(1)` space, using
/// the symmetry `C(n, k) == C(n, n - k)` and an exact division at every
/// step to avoid intermediate overflow.
///
/// # Panics
///
/// Panics if `C(n, k)` does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::combinations::combinations;
///
/// assert_eq!(combinations(5, 2), 10);
/// assert_eq!(combinations(5, 6), 0);
/// ```
pub fn combinations(n: u64, k: u64) -> u128 {
    let n = n as u128;
    let mut k = k as u128;

    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    if k > n / 2 {
        k = n - k;
    }

    let mut result = 1u128;
    for i in 1..=k {
        result = mul_div(result, n - i + 1, i);
    }
    result
}
