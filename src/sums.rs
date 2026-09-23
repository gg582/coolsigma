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

fn simplex_number_wide(n: u128, dimension: u128) -> u128 {
    // C(n + dimension, dimension + 1) built up as C(n + i - 1, i) with an
    // exact division at every step.
    let mut result = 1u128;
    for i in 1..=dimension + 1 {
        result = mul_div(result, n + i - 1, i);
    }
    result
}

/// The `n`-th simplex number of the given dimension: `C(n + dimension, dimension + 1)`.
///
/// `dimension = 0` gives `n` itself, `dimension = 1` the triangular numbers
/// `0, 1, 3, 6, 10, ...`, `dimension = 2` the tetrahedral numbers
/// `0, 1, 4, 10, 20, ...`, and `dimension = 3` the pentatope numbers.
///
/// Runs in `O(dimension)` time with `O(1)` space.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::simplex_number;
///
/// assert_eq!(simplex_number(5, 1), 15); // triangular
/// assert_eq!(simplex_number(5, 2), 35); // tetrahedral
/// assert_eq!(simplex_number(5, 3), 70); // pentatope
/// ```
pub fn simplex_number(n: u64, dimension: u64) -> u128 {
    simplex_number_wide(n as u128, dimension as u128)
}

/// Sum of `count` consecutive simplex numbers starting at index `start`:
///
/// ```text
/// sum_{k=start}^{start + count - 1} simplex_number(k, dimension)
///     == simplex_number(start + count - 1, dimension + 1)
///      - simplex_number(start - 1, dimension + 1)
/// ```
///
/// This is the segment decomposition (分積法) generalized to any order:
/// `dimension = 0` sums plain integers, `dimension = 1` the triangular
/// numbers, `dimension = 2` the tetrahedral numbers, and so on.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::segment_simplex_sum;
///
/// assert_eq!(segment_simplex_sum(4, 5, 0), 30); // 4 + 5 + 6 + 7 + 8
/// assert_eq!(segment_simplex_sum(4, 5, 1), 110); // T(4) + ... + T(8)
/// assert_eq!(segment_simplex_sum(4, 5, 2), 315); // S2(4) + ... + S2(8)
/// ```
pub fn segment_simplex_sum(start: u64, count: u64, dimension: u64) -> u128 {
    if count == 0 {
        return 0;
    }

    let end = start as u128 + count as u128 - 1;
    let dimension = dimension as u128 + 1;

    // simplex_number(x, dimension) is non-decreasing in x, so `hi >= lo`
    // and the subtraction cannot underflow.
    let hi = simplex_number_wide(end, dimension);
    let lo = if start == 0 {
        0
    } else {
        simplex_number_wide(start as u128 - 1, dimension)
    };
    hi - lo
}
