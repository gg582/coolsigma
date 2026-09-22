//! Closed-form sums of polynomial series.

use crate::common::mul_div;

/// Generalized combinatorial series sum
/// `prod_{i=0}^{r} (n + i) * ((r + 1) * n + 1) / (r + 2)!`.
///
/// For `r = 0` this is the sum of the first `n` integers and for `r = 1` the
/// sum of the first `n` squares. All arithmetic is done in `u128` with
/// factor cancellation to avoid intermediate overflow.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::generalized_series_sum;
///
/// assert_eq!(generalized_series_sum(5, 1), 55); // 1^2 + 2^2 + ... + 5^2
/// ```
pub fn generalized_series_sum(n: u64, r: u64) -> u128 {
    let n = n as u128;
    let r = r as u128;

    // prod_{i=0}^{r} (n + i) / (r + 1)! == C(n + r, r + 1), built up with an
    // exact division at every step.
    let mut result = 1u128;
    for i in 0..=r {
        result = mul_div(result, n + i, i + 1);
    }

    // prod_{i=0}^{r} (n + i) == C(n + r, r + 1) * (r + 1)!, so the remaining
    // factor of (r + 2)! leaves a single (r + 2) in the denominator.
    mul_div(result, (r + 1) * n + 1, r + 2)
}

/// Sum of an arithmetic sequence of `count` terms starting at `first` with
/// common difference `difference`:
///
/// ```text
/// sum_{i=0}^{count - 1} (first + i * difference)
///     == count * (2 * first + (count - 1) * difference) / 2
/// ```
///
/// With `difference = 1` this reduces to a segment sum of consecutive
/// integers, `sum_{k=first}^{first + count - 1} k`, the classical
/// "triangle + parallelogram" discretization: `count * (count + 1) / 2`
/// plus `(first - 1) * count`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::arithmetic_series_sum;
///
/// assert_eq!(arithmetic_series_sum(4, 4, 1), 22); // 4 + 5 + 6 + 7
/// assert_eq!(arithmetic_series_sum(1, 5, 2), 25); // 1 + 3 + 5 + 7 + 9
/// ```
pub fn arithmetic_series_sum(first: u64, count: u64, difference: u64) -> u128 {
    if count == 0 {
        return 0;
    }

    let first = first as u128;
    let count = count as u128;
    let difference = difference as u128;

    // 2 * first + (count - 1) * difference always fits: both addends are at
    // most ~2^65 and ~2^128 - 2^65 respectively, and mul_div cancels the
    // trailing / 2 against an even factor before multiplying.
    let pair_sum = 2 * first + (count - 1) * difference;

    mul_div(count, pair_sum, 2)
}

/// Centered polygonal expansion number: dots arranged in `layer` nested
/// polygonal layers around a single center point,
/// `1 + sides * (1 + 2 + ... + layer)`.
///
/// `sides = 6` gives the centered hexagonal ("circular expansion") numbers
/// `1, 7, 19, 37, ...` and `sides = 8` the odd squares ("square expansion")
/// `(2 * layer + 1)^2 = 1, 9, 25, 49, ...`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::centered_expansion;
///
/// assert_eq!(centered_expansion(6, 2), 19); // 1 + 6 * (1 + 2)
/// assert_eq!(centered_expansion(8, 2), 25); // (2 * 2 + 1)^2
/// ```
pub fn centered_expansion(sides: u64, layer: u64) -> u128 {
    let sides = sides as u128;
    let layer = layer as u128;

    // 1 + sides * layer * (layer + 1) / 2
    1 + mul_div(layer, sides * (layer + 1), 2)
}

/// Sum of squares iterated `depth` times:
/// `sum_{k=1}^{n} ... sum_{k=1}^{n} k^2` with `depth` summation signs,
/// equal to `prod_{i=0}^{depth} (n + i) * (2 * n + depth) / (depth + 2)!`.
///
/// `depth = 1` gives the sum of the first `n` squares.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::iterated_square_sum;
///
/// assert_eq!(iterated_square_sum(5, 2), 105);
/// ```
pub fn iterated_square_sum(n: u64, depth: u64) -> u128 {
    let n = n as u128;
    let depth = depth as u128;

    // prod_{i=0}^{depth} (n + i) / (depth + 1)! == C(n + depth, depth + 1).
    let mut result = 1u128;
    for i in 0..=depth {
        result = mul_div(result, n + i, i + 1);
    }

    mul_div(result, 2 * n + depth, depth + 2)
}

/// Weighted partial-square sum `sum_{k=1}^{n} k * (1^2 + ... + k^2)`,
/// equal to `n * (n + 1) * (n + 2) * (8 * n^2 + 11 * n + 1) / 120`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::weighted_partial_square_sum;
///
/// assert_eq!(weighted_partial_square_sum(5), 448);
/// ```
pub fn weighted_partial_square_sum(n: u64) -> u128 {
    let n = n as u128;

    // C(n + 2, 3) with an exact division at every step.
    let mut triple = 1u128;
    for i in 0..=2u128 {
        triple = mul_div(triple, n + i, i + 1);
    }

    let quad = 8u128
        .checked_mul(n * n)
        .and_then(|v| v.checked_add(11 * n + 1))
        .expect("weighted_partial_square_sum: result exceeds u128");

    mul_div(triple, quad, 20)
}

/// Sum of the interleaved sequence of centered hexagonal numbers and triple
/// squares `1, 3, 7, 12, 19, 27, 37, ...`.
///
/// Runs in `O(1)` time via the closed form
/// `ceil(n/2)^3 + floor(n/2) * (floor(n/2) + 1) * (2 * floor(n/2) + 1) / 2`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::interleaved_series_sum;
///
/// assert_eq!(interleaved_series_sum(5), 42); // 1 + 3 + 7 + 12 + 19
/// ```
pub fn interleaved_series_sum(n: u64) -> u128 {
    let n = n as u128;
    let upper = n.div_ceil(2);
    let lower = n / 2;

    let cube = upper
        .checked_mul(upper)
        .and_then(|v| v.checked_mul(upper))
        .expect("interleaved_series_sum: result exceeds u128");

    // (lower * (lower + 1) / 2) * (2 * lower + 1): the first factor is exact.
    let even_part = lower * (lower + 1) / 2 * (2 * lower + 1);

    cube + even_part
}
