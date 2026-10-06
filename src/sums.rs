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

/// [`segment_simplex_sum`] evaluated through the Chu–Vandermonde
/// convolution instead of the hockey-stick difference.
///
/// Write `M(x, k) = C(x + k - 1, k)` for the multiset coefficient (see
/// [`multiset_coefficient`](crate::combinations::multiset_coefficient)), so
/// the summand is `simplex_number(k, dimension) == M(k, r)` with
/// `r = dimension + 1`. With `a = start - 1` and `b = count`, the
/// hockey-stick identity gives `M(a + b, r + 1) - M(a, r + 1)`, and the
/// Vandermonde identity for multiset coefficients
/// `M(a + b, K) = sum_{j=0}^{K} M(a, j) M(b, K - j)` expands the first term.
/// Its `j = r + 1` term cancels `M(a, r + 1)`, leaving
///
/// ```text
/// sum_{k=start}^{start + count - 1} M(k, r)
///     == sum_{j=0}^{r} M(start - 1, j) * M(count, r + 1 - j)
/// ```
///
/// which separates the offset (`start - 1`) from the segment length
/// (`count`). Each product is the contribution of a fixed offset order `j`.
///
/// The result is always equal to `segment_simplex_sum(start, count,
/// dimension)`. Runs in `O(dimension)` time.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::{segment_simplex_sum, segment_simplex_sum_vandermonde};
///
/// // T(4) + ... + T(8) = M(5,3) + M(3,1) M(5,2) + M(3,2) M(5,1)
/// //                   = 35 + 3 * 15 + 6 * 5 = 110
/// assert_eq!(segment_simplex_sum_vandermonde(4, 5, 1), 110);
/// assert_eq!(
///     segment_simplex_sum_vandermonde(4, 5, 2),
///     segment_simplex_sum(4, 5, 2),
/// );
/// ```
pub fn segment_simplex_sum_vandermonde(start: u64, count: u64, dimension: u64) -> u128 {
    if count == 0 {
        return 0;
    }
    if start == 0 {
        // simplex_number(0, dimension) == 0, so drop the leading term to keep
        // the offset `start - 1` non-negative.
        return segment_simplex_sum_vandermonde(1, count - 1, dimension);
    }

    let a = start as u128 - 1;
    let b = count as u128;
    let order = dimension as u128 + 1;

    // Walk j upward, updating both factors by exact ratios:
    //   M(a, j + 1) = M(a, j) * (a + j) / (j + 1)
    //   M(b, t - 1) = M(b, t) * t / (b + t - 1)
    let mut offset_term = 1u128; // M(a, 0)
    let mut length_term = simplex_number_wide(b, order); // M(b, order + 1)
    let mut total = 0u128;
    for j in 0..=order {
        let product = offset_term
            .checked_mul(length_term)
            .expect("segment_simplex_sum_vandermonde: result exceeds u128");
        total = total
            .checked_add(product)
            .expect("segment_simplex_sum_vandermonde: result exceeds u128");

        if j < order {
            offset_term = mul_div(offset_term, a + j, j + 1);
            let t = order + 1 - j;
            length_term = mul_div(length_term, t, b + t - 1);
        }
    }
    total
}

/// Linearly weighted segment sum of simplex numbers: the first term gets
/// weight `count`, the next `count - 1`, down to weight `1` on the last.
///
/// ```text
/// sum_{i=0}^{count-1} (count - i) * simplex_number(start + i, dimension)
/// ```
///
/// Swapping the order of summation turns this into a sum of partial sums:
/// `sum_{i=1}^{count} sum_{j=1}^{i} a_j`, where `a_j` is the `j`-th term of
/// the segment. Each partial sum telescopes by the hockey-stick identity,
/// which gives the closed form
///
/// ```text
/// segment_simplex_sum(start, count, dimension + 1)
///     - count * simplex_number(start - 1, dimension + 1)
/// ```
///
/// For `start = 1` this reduces to `simplex_number(count, dimension + 2)`.
///
/// # Panics
///
/// Panics if the unweighted outer sum
/// `segment_simplex_sum(start, count, dimension + 1)` does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::weighted_segment_simplex_sum;
///
/// // 3*T(2) + 2*T(3) + 1*T(4) = 9 + 12 + 10
/// assert_eq!(weighted_segment_simplex_sum(2, 3, 1), 31);
/// ```
pub fn weighted_segment_simplex_sum(start: u64, count: u64, dimension: u64) -> u128 {
    if count == 0 {
        return 0;
    }

    let outer = segment_simplex_sum(start, count, dimension + 1);
    let offset = if start == 0 {
        0
    } else {
        simplex_number_wide(start as u128 - 1, dimension as u128 + 1)
    };
    // outer == weighted + count * offset, so this cannot underflow.
    outer - count as u128 * offset
}
