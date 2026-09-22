//! Sums over simplex (binomial-coefficient) sequences.

use crate::common::mul_div;

/// Generalized simplex sum, equal to the binomial coefficient `C(n + k, n + 1)`.
///
/// For `k = 1` this is the `n`-th triangular number, for `k = 2` the `n`-th
/// tetrahedral number, and so on.
///
/// Runs in `O(n)` time with `O(1)` space.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::simplex_sum;
///
/// assert_eq!(simplex_sum(1, 10), 55); // 1 + 2 + ... + 10
/// assert_eq!(simplex_sum(2, 10), 220);
/// ```
pub fn simplex_sum(n: u64, k: u64) -> u128 {
    if k == 0 {
        return 0;
    }

    let n = n as u128;
    let k = k as u128;

    let mut result = 1u128;
    for i in 1..=n + 1 {
        result = mul_div(result, n + k - i + 1, i);
    }
    result
}

/// Simplex sum truncated to the terms whose second index runs from `start`
/// to `k`, i.e. `simplex_sum(n, k) - simplex_sum(n, start - 1)`.
///
/// Returns `0` when `start == 0`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::truncated_simplex_sum;
///
/// assert_eq!(truncated_simplex_sum(4, 1, 6), 15); // 4 + 5 + 6
/// ```
pub fn truncated_simplex_sum(start: u64, n: u64, k: u64) -> u128 {
    if start == 0 {
        return 0;
    }
    simplex_sum(n, k).saturating_sub(simplex_sum(n, start - 1))
}
