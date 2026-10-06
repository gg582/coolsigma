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

/// Multiset coefficient `((n multichoose k)) = C(n + k - 1, k)`: the number
/// of size-`k` multisets drawn from `n` kinds.
///
/// Equivalently the rising factorial `n (n + 1) ... (n + k - 1) / k!`. For
/// fixed `k` this is the figurate sequence of order `k`: `k = 1` gives `n`,
/// `k = 2` the triangular numbers, `k = 3` the tetrahedral numbers, so
/// `multiset_coefficient(n, k) == simplex_number(n, k - 1)` for `k >= 1`.
///
/// Returns `1` when `k == 0` (including `n == 0`) and `0` when `n == 0 < k`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::combinations::multiset_coefficient;
///
/// assert_eq!(multiset_coefficient(4, 0), 1);
/// assert_eq!(multiset_coefficient(4, 2), 10); // triangular
/// assert_eq!(multiset_coefficient(4, 3), 20); // tetrahedral
/// assert_eq!(multiset_coefficient(0, 3), 0);
/// ```
pub fn multiset_coefficient(n: u64, k: u64) -> u128 {
    if k == 0 {
        return 1;
    }
    if n == 0 {
        return 0;
    }

    // C(n + k - 1, k) == C(n + k - 1, n - 1); iterate over the smaller one.
    let top = n as u128 + k as u128 - 1;
    let k = (k as u128).min(n as u128 - 1);

    let mut result = 1u128;
    for i in 1..=k {
        result = mul_div(result, top - k + i, i);
    }
    result
}
