//! Sums over simplex (binomial-coefficient) sequences.

use crate::common::{mul_div, multiset, simplex};
use crate::series::generalized_series_sum;

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
    // C(n + k, n + 1) is the k-th simplex number of dimension n.
    simplex(k as u128, n as u128)
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

/// The `n`-th simplex number of the given dimension: `C(n + dimension, dimension + 1)`.
///
/// Successive dimensions are related by the ratio recurrence
/// `simplex_number(n, d + 1) == simplex_number(n, d) * (n + d + 1) / (d + 2)`,
/// which is how the value is built up from `simplex_number(n, 0) == n`.
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
    simplex(n as u128, dimension as u128)
}

/// Sum of `count` consecutive simplex numbers starting at index `start`:
///
/// ```text
/// sum_{k=start}^{start + count - 1} simplex_number(k, dimension)
///     == simplex_number(start + count - 1, dimension + 1)
///      - simplex_number(start - 1, dimension + 1)
/// ```
///
/// This is the segment of a figurate sequence of any order:
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
    let hi = simplex(end, dimension);
    let lo = if start == 0 {
        0
    } else {
        simplex(start as u128 - 1, dimension)
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
    iterated_segment_simplex_sum(start, count, dimension, 1)
}

/// `depth`-fold iterated partial sum of a segment of simplex numbers.
///
/// Let `a_i = simplex_number(start + i, dimension)` for `i = 0, ..., count - 1`.
/// Summing the segment once gives [`segment_simplex_sum`]; summing the running
/// totals again gives [`weighted_segment_simplex_sum`]; in general, swapping
/// the order of summation turns `depth` nested sums into one weighted sum
/// whose weights are multiset coefficients `M(x, p) = C(x + p - 1, p)`:
///
/// ```text
/// sum_{i=0}^{count-1} M(count - i, depth - 1) * a_i
/// ```
///
/// (`depth = 2` gives the linear weights `count, count - 1, ..., 1`.)
///
/// With `r = dimension + 1` the summand is `a_i = M(start + i, r)`. The
/// Chu–Vandermonde convolution separates the offset `start - 1` from the
/// segment length, and each further summation over the length raises the
/// length factor's order by one (`sum_{x=1}^{c} M(x, t) == M(c, t + 1)`):
///
/// ```text
/// sum_{j=0}^{r} M(start - 1, j) * M(count, r + depth - j)
/// ```
///
/// For `start = 1` only `j = 0` survives and the result is
/// `simplex_number(count, dimension + depth)`; in particular the classical
/// identity `simplex_number(n, d) == sum_{k=1}^{n} (n - k + 1) *
/// simplex_number(k, d - 2)` is the case `depth = 2`.
///
/// `depth = 0` means no summation and returns the last term of the segment
/// (or `0` for an empty segment). Runs in `O(dimension + depth)` time.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::{
///     iterated_segment_simplex_sum, segment_simplex_sum, simplex_number,
///     weighted_segment_simplex_sum,
/// };
///
/// assert_eq!(iterated_segment_simplex_sum(4, 5, 1, 1), segment_simplex_sum(4, 5, 1));
/// assert_eq!(iterated_segment_simplex_sum(2, 3, 1, 2), weighted_segment_simplex_sum(2, 3, 1));
///
/// // 10*T(2) + 6*T(3) + 3*T(4) + 1*T(5) = 30 + 36 + 30 + 15
/// assert_eq!(iterated_segment_simplex_sum(2, 4, 1, 3), 111);
///
/// // sum_{k=1}^{n} (n - k + 1) * simplex_number(k, 3) == simplex_number(n, 5)
/// assert_eq!(iterated_segment_simplex_sum(1, 7, 3, 2), simplex_number(7, 5));
/// ```
pub fn iterated_segment_simplex_sum(start: u64, count: u64, dimension: u64, depth: u64) -> u128 {
    if count == 0 {
        return 0;
    }
    if depth == 0 {
        return simplex(start as u128 + count as u128 - 1, dimension as u128);
    }
    if start == 0 {
        // simplex_number(0, dimension) == 0, so drop the leading term; the
        // remaining weights are exactly those of a segment one shorter.
        return iterated_segment_simplex_sum(1, count - 1, dimension, depth);
    }

    let a = start as u128 - 1;
    let b = count as u128;
    let order = dimension as u128 + 1;
    let depth = depth as u128;

    // Walk j upward, updating both factors by exact ratios:
    //   M(a, j + 1) = M(a, j) * (a + j) / (j + 1)
    //   M(b, t - 1) = M(b, t) * t / (b + t - 1)
    let mut offset_term = 1u128; // M(a, 0)
    let mut length_term = multiset(b, order + depth); // M(b, order + depth)
    let mut total = 0u128;
    for j in 0..=order {
        let product = offset_term
            .checked_mul(length_term)
            .expect("iterated_segment_simplex_sum: result exceeds u128");
        total = total
            .checked_add(product)
            .expect("iterated_segment_simplex_sum: result exceeds u128");

        if j < order {
            offset_term = mul_div(offset_term, a + j, j + 1);
            let t = order + depth - j;
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
        simplex(start as u128 - 1, dimension as u128 + 1)
    };
    // outer == weighted + count * offset, so this cannot underflow.
    outer - count as u128 * offset
}

/// Convolution of two simplex sequences:
///
/// ```text
/// sum_{k=1}^{n} simplex_number(n - k + 1, a) * simplex_number(k, b)
///     == simplex_number(n, a + b + 2)
/// ```
///
/// Both factors are coefficients of `z / (1 - z)^(d + 1)` (with `d` the
/// dimension), so the convolution is the coefficient of
/// `z^(n+1)` in `z^2 / (1 - z)^(a + b + 2)`. With `a = 0` the left factor is
/// the linear weight `n - k + 1`, recovering
/// `sum_k (n - k + 1) * simplex_number(k, b) == simplex_number(n, b + 2)`;
/// with `a = b = 1` the products of triangular numbers
/// `T(n) T(1) + T(n - 1) T(2) + ... + T(1) T(n)` sum to `simplex_number(n, 4)`.
///
/// # Panics
///
/// Panics if `a + b + 2` overflows `u64` or the result does not fit in
/// `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::simplex_convolution;
///
/// // T(4) T(1) + T(3) T(2) + T(2) T(3) + T(1) T(4) = 10 + 18 + 18 + 10
/// assert_eq!(simplex_convolution(4, 1, 1), 56);
/// ```
pub fn simplex_convolution(n: u64, a: u64, b: u64) -> u128 {
    let dimension = a
        .checked_add(b)
        .and_then(|d| d.checked_add(2))
        .expect("simplex_convolution: dimension exceeds u64");
    simplex(n as u128, dimension as u128)
}

/// `sum_{j=0}^{last} M(p, j) * generalized_series_sum(n, order - j)`: the
/// Chu–Vandermonde split of index-weighted segment sums into an offset part
/// and a length part.
fn offset_index_weighted_convolution(p: u64, n: u64, order: u64, last: u64) -> u128 {
    let overflow = "index-weighted segment sum: result exceeds u128";
    let mut offset_term = 1u128; // M(p, 0)
    let mut total = 0u128;
    for j in 0..=last {
        if offset_term == 0 {
            break; // M(0, j) == 0 for every j >= 1
        }
        let product = offset_term
            .checked_mul(generalized_series_sum(n, order - j))
            .expect(overflow);
        total = total.checked_add(product).expect(overflow);
        if j < last {
            offset_term = mul_div(offset_term, p as u128 + j as u128, j as u128 + 1);
        }
    }
    total
}

/// Segment of simplex numbers weighted by position, ascending:
///
/// ```text
/// sum_{i=0}^{count-1} (i + 1) * simplex_number(start + i, dimension)
/// ```
///
/// (the mirror image of [`weighted_segment_simplex_sum`], whose weights
/// descend). With `r = dimension + 1`, `p = start - 1`,
/// `M(x, t) = C(x + t - 1, t)` and `G(n, t) = generalized_series_sum(n, t)
/// = sum_{k=1}^{n} k * M(k, t)`, the Chu–Vandermonde convolution
/// `M(p + k, r) = sum_j M(p, j) M(k, r - j)` gives
///
/// ```text
/// sum_{j=0}^{r} M(p, j) * G(count, r - j)
/// ```
///
/// For triangular numbers (`dimension = 1`) this is
/// `G(n, 2) + p * G(n, 1) + T(p) * n (n + 1) / 2`, i.e.
/// `G(n, 2) + 2p * S(n, 2) + (T(p) - p) * S(n, 1)`. It can also be written
/// as `T(start) * S(n, 1) + 2 (start + 1) * S(n - 1, 2) + 3 * S(n - 2, 3)`,
/// the Newton forward-difference form.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::sums::ascending_weighted_segment_simplex_sum;
///
/// // 1*T(3) + 2*T(4) + 3*T(5) = 6 + 20 + 45
/// assert_eq!(ascending_weighted_segment_simplex_sum(3, 3, 1), 71);
/// ```
pub fn ascending_weighted_segment_simplex_sum(start: u64, count: u64, dimension: u64) -> u128 {
    if count == 0 {
        return 0;
    }
    if start == 0 {
        // Drop the zero leading term: the remaining weights are one larger
        // than those of a segment starting at 1, hence the extra plain sum.
        return ascending_weighted_segment_simplex_sum(1, count - 1, dimension)
            .checked_add(segment_simplex_sum(1, count - 1, dimension))
            .expect("ascending_weighted_segment_simplex_sum: result exceeds u128");
    }

    let order = dimension + 1;
    offset_index_weighted_convolution(start - 1, count, order, order)
}

/// Segment of simplex numbers weighted by trapezoidal numbers: the term at
/// position `i` gets weight `(i + 1) + (i + 2) + ... + count`.
///
/// ```text
/// sum_{i=0}^{count-1} simplex_number(start + i, dimension) * sum_{k=i+1}^{count} k
/// ```
///
/// Swapping the order of summation shows this is the index-weighted sum of
/// running segment totals, `sum_{k=1}^{count} k * segment_simplex_sum(start,
/// k, dimension)`. With `r = dimension + 1`, `p = start - 1` and `M`, `G` as
/// in [`ascending_weighted_segment_simplex_sum`], the Chu–Vandermonde split
/// leaves
///
/// ```text
/// sum_{j=0}^{r} M(p, j) * G(count, r + 1 - j)
/// ```
///
/// For `start = 1` this is `generalized_series_sum(count, r + 1)`; for
/// `dimension = 0` it is `G(n, 2) + (start - 1) * G(n, 1)`, the sum of each
/// index times its triangular number plus `start - 1` times the sum of
/// squares. In Newton forward-difference form (`dimension = 0`) it is
/// `start * S(n, 1) + 2 (start + 1) * S(n - 1, 2) + 3 * S(n - 2, 3)`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::generalized_series_sum;
/// use coolsigma::sums::trapezoid_weighted_segment_simplex_sum;
///
/// // 3*(1+2+3) + 4*(2+3) + 5*3 = 18 + 20 + 15
/// assert_eq!(trapezoid_weighted_segment_simplex_sum(3, 3, 0), 53);
/// // 1*T(n) + 2*(T(n) - T(1)) + ... == sum_k k * T(k)
/// assert_eq!(trapezoid_weighted_segment_simplex_sum(1, 6, 0), generalized_series_sum(6, 2));
/// ```
pub fn trapezoid_weighted_segment_simplex_sum(start: u64, count: u64, dimension: u64) -> u128 {
    if count == 0 {
        return 0;
    }
    if start == 0 {
        // Drop the zero leading term: the remaining trapezoidal weights are
        // those of a segment one shorter plus the descending weights
        // `count - 1, ..., 1`.
        return trapezoid_weighted_segment_simplex_sum(1, count - 1, dimension)
            .checked_add(weighted_segment_simplex_sum(1, count - 1, dimension))
            .expect("trapezoid_weighted_segment_simplex_sum: result exceeds u128");
    }

    let order = dimension + 1;
    offset_index_weighted_convolution(start - 1, count, order + 1, order)
}
