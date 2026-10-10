//! Closed-form sums of polynomial series.

use crate::common::{gcd, mul_div, mul_div_rem, multiset, newton_sum, power_differences, simplex};

/// Generalized combinatorial series sum
/// `prod_{i=0}^{r} (n + i) * ((r + 1) * n + 1) / (r + 2)!`.
///
/// This is the index-weighted sum of a figurate sequence: with the multiset
/// coefficient `M(k, r) = C(k + r - 1, r)` (so `M(k, r + 1)` is the simplex
/// number `simplex_number(k, r)`),
///
/// ```text
/// generalized_series_sum(n, r) == sum_{k=1}^{n} k * M(k, r)
///                              == simplex_number(n, r + 1) * ((r + 1) * n + 1) / (r + 2)
/// ```
///
/// For `r = 0` this is the sum of the first `n` integers, for `r = 1` the
/// sum of the first `n` squares, and for `r = 2` the sum
/// `sum_{k=1}^{n} k * T(k)` of each index times its triangular number,
/// `n (n + 1) (n + 2) (3n + 1) / 24`.
///
/// From `k * M(k, r) == (r + 1) * M(k - 1, r + 1) + M(k, r)` it also equals
///
/// ```text
/// (r + 1) * simplex_number(n - 1, r + 1) + simplex_number(n, r)
///     == r * simplex_number(n - 1, r + 1) + simplex_number(n, r + 1)
/// ```
///
/// (`r = 2`: `S(n, 3) + 2 S(n - 1, 3)`). Swapping the order of summation in
/// `sum_k k * sum_{j<=k} M(j, r - 1)` gives the trapezoidal-number form
/// `sum_{j=1}^{n} M(j, r - 1) * (j + (j + 1) + ... + n)`; see
/// [`trapezoid_weighted_segment_simplex_sum`](crate::sums::trapezoid_weighted_segment_simplex_sum).
///
/// All arithmetic is done in `u128` with factor cancellation to avoid
/// intermediate overflow.
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
/// assert_eq!(generalized_series_sum(4, 2), 65); // 1*1 + 2*3 + 3*6 + 4*10
/// ```
pub fn generalized_series_sum(n: u64, r: u64) -> u128 {
    let n = n as u128;
    let r = r as u128;

    // prod_{i=0}^{r} (n + i) / (r + 1)! == C(n + r, r + 1).
    let result = simplex(n, r);

    // prod_{i=0}^{r} (n + i) == C(n + r, r + 1) * (r + 1)!, so the remaining
    // factor of (r + 2)! leaves a single (r + 2) in the denominator.
    mul_div(result, (r + 1) * n + 1, r + 2)
}

/// Segment of the index-weighted figurate sum:
///
/// ```text
/// sum_{k=start}^{start + count - 1} k * M(k, r)
///     == generalized_series_sum(start + count - 1, r) - generalized_series_sum(start - 1, r)
/// ```
///
/// with `M(k, r) = C(k + r - 1, r)`. For `r = 2` this is
/// `sum_k k * T(k)` over a segment of triangular numbers. Writing
/// `k = (start - 1) + (k - start + 1)` splits it into an offset multiple of
/// the plain segment sum plus a sum weighted by the position in the segment
/// (for `r >= 1`):
///
/// ```text
/// (start - 1) * segment_simplex_sum(start, count, r - 1)
///     + ascending_weighted_segment_simplex_sum(start, count, r - 1)
/// ```
///
/// (see [`ascending_weighted_segment_simplex_sum`](crate::sums::ascending_weighted_segment_simplex_sum)).
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::segment_generalized_series_sum;
///
/// assert_eq!(segment_generalized_series_sum(3, 3, 2), 18 + 40 + 75); // 3*6 + 4*10 + 5*15
/// ```
pub fn segment_generalized_series_sum(start: u64, count: u64, r: u64) -> u128 {
    if count == 0 {
        return 0;
    }

    let end = start
        .checked_add(count - 1)
        .expect("segment_generalized_series_sum: result exceeds u128");
    // generalized_series_sum(x, r) is non-decreasing in x.
    let hi = generalized_series_sum(end, r);
    let lo = if start == 0 {
        0
    } else {
        generalized_series_sum(start - 1, r)
    };
    hi - lo
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
/// `depth = 0` gives `n^2`, `depth = 1` the sum of the first `n` squares
/// (square pyramidal numbers), `depth = 2` their partial sums, and so on.
/// Equivalently the iterated sum is a rescaled simplex number,
///
/// ```text
/// iterated_square_sum(n, depth)
///     == simplex_number(n, depth) * (2 * n + depth) / (depth + 2)
/// ```
///
/// and, swapping the order of summation, the linearly weighted sum
/// `sum_{k=1}^{n} (n - k + 1) * iterated_square_sum(k, depth - 2)`.
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
    mul_div(simplex(n, depth), 2 * n + depth, depth + 2)
}

/// Number of unit cubes in a stepped rectangular frustum: `layers` stacked
/// rectangles, the top one `top_length x top_width`, each lower layer one
/// unit longer in both directions.
///
/// ```text
/// sum_{i=0}^{layers-1} (top_length + i) * (top_width + i)
/// ```
///
/// Evaluated by the discrete prismoidal formula. With top `a x b` and bottom
/// `c x d` (`c = a + layers - 1`, `d = b + layers - 1`),
///
/// ```text
/// layers * [a * (2b + d) + c * (2d + b) + (c - a)] / 6
/// ```
///
/// i.e. the continuous prismoid volume `h/6 * [(2b + d)a + (2d + b)c]` plus
/// the correction `h/6 * (c - a)` for the discrete layers. A square top
/// (`top_length == top_width == m`) gives the segment sum of squares
/// `m^2 + (m + 1)^2 + ... + (m + layers - 1)^2`, and `m = 1` the square
/// pyramidal numbers.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::rectangular_frustum_sum;
///
/// assert_eq!(rectangular_frustum_sum(1, 1, 5), 55); // 1 + 4 + 9 + 16 + 25
/// assert_eq!(rectangular_frustum_sum(3, 3, 3), 50); // 9 + 16 + 25
/// assert_eq!(rectangular_frustum_sum(2, 4, 3), 47); // 2*4 + 3*5 + 4*6
/// ```
pub fn rectangular_frustum_sum(top_length: u64, top_width: u64, layers: u64) -> u128 {
    if layers == 0 {
        return 0;
    }

    let (a, b, n) = (top_length as u128, top_width as u128, layers as u128);
    let (c, d) = (a + n - 1, b + n - 1);

    // n * bracket is a multiple of 6, so after taking g = gcd(n, 6) out of n
    // the bracket is a multiple of h = 6 / g. Divide each term by h before
    // adding so no intermediate value exceeds the result.
    let g = gcd(n, 6);
    let h = 6 / g;
    let (q1, r1) = mul_div_rem(a, 2 * b + d, h);
    let (q2, r2) = mul_div_rem(c, 2 * d + b, h);
    let (q3, r3) = ((c - a) / h, (c - a) % h);

    q1.checked_add(q2)
        .and_then(|v| v.checked_add(q3 + (r1 + r2 + r3) / h))
        .and_then(|v| v.checked_mul(n / g))
        .expect("rectangular_frustum_sum: result exceeds u128")
}

/// Sum of `count` consecutive iterated square sums starting at index `start`:
///
/// ```text
/// sum_{k=start}^{start + count - 1} iterated_square_sum(k, order)
/// ```
///
/// Each summation raises the order by one, so this telescopes to
/// `iterated_square_sum(start + count - 1, order + 1)
/// - iterated_square_sum(start - 1, order + 1)`. For `order = 0` (a segment
/// of plain squares) the value is [`rectangular_frustum_sum`] with a square
/// top of side `start`.
///
/// This is [`iterated_segment_iterated_square_sum`] with `depth = 1`; see
/// [`segment_iterated_square_sum_offset_expansion`] for the same value
/// expanded in powers of the offset `start - 1`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::segment_iterated_square_sum;
///
/// assert_eq!(segment_iterated_square_sum(3, 3, 0), 50); // 9 + 16 + 25
/// assert_eq!(segment_iterated_square_sum(3, 3, 1), 99); // 14 + 30 + 55
/// ```
pub fn segment_iterated_square_sum(start: u64, count: u64, order: u64) -> u128 {
    iterated_segment_iterated_square_sum(start, count, order, 1)
}

/// [`segment_iterated_square_sum`] expanded in the offset `p = start - 1`.
///
/// This is [`iterated_segment_iterated_square_sum_offset_expansion`] with
/// `depth = 1`:
///
/// ```text
/// sum_{k=p+1}^{p+count} Q(k, order)
///     == Q(count, r) + 2p * S(count, r) + sum_{j=0}^{r-1} Q(p, j) * S(count, r - 1 - j)
/// ```
///
/// with `r = order + 1`, `S(n, q) = simplex_number(n, q)` and
/// `Q(n, q) = iterated_square_sum(n, q)`. For `order = 0` this is
/// `Q(n, 1) + 2p * S(n, 1) + p^2 * n`; for `order = 1` it is
/// `Q(n, 2) + 2p * S(n, 2) + p^2 * S(n, 1) + Q(p, 1) * n`.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::{
///     segment_iterated_square_sum, segment_iterated_square_sum_offset_expansion,
/// };
///
/// // Q(3,2) + 4 S(3,2) + 4 S(3,1) + 5 * 3 = 20 + 40 + 24 + 15
/// assert_eq!(segment_iterated_square_sum_offset_expansion(3, 3, 1), 99);
/// assert_eq!(
///     segment_iterated_square_sum_offset_expansion(4, 6, 3),
///     segment_iterated_square_sum(4, 6, 3),
/// );
/// ```
pub fn segment_iterated_square_sum_offset_expansion(start: u64, count: u64, order: u64) -> u128 {
    iterated_segment_iterated_square_sum_offset_expansion(start, count, order, 1)
}

/// `depth`-fold iterated partial sum of a segment of iterated square sums.
///
/// Let `Q(n, q) = iterated_square_sum(n, q)`, `M(x, t) = C(x + t - 1, t)` and
/// `a_i = Q(start + i, order)` for `i = 0, ..., count - 1`. Summing the
/// segment `depth` times and swapping the order of summation gives
///
/// ```text
/// sum_{i=0}^{count-1} M(count - i, depth - 1) * Q(start + i, order)
/// ```
///
/// so `depth = 1` is the plain segment sum ([`segment_iterated_square_sum`])
/// and `depth = 2` weights the terms `count, count - 1, ..., 1`. With
/// `p = start - 1`, repeated telescoping gives the closed form
///
/// ```text
/// Q(p + count, order + depth) - sum_{t=0}^{depth-1} M(count, t) * Q(p, order + depth - t)
/// ```
///
/// For `start = 1` this is `Q(count, order + depth)`; in particular
/// `Q(n, r) == sum_{k=1}^{n} (n - k + 1) * Q(k, r - 2)`.
///
/// `depth = 0` returns the last term of the segment (or `0` when empty). See
/// [`iterated_segment_iterated_square_sum_offset_expansion`] for the
/// expansion in powers of the offset.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::{iterated_segment_iterated_square_sum, iterated_square_sum};
///
/// // 3*Q(3,1) + 2*Q(4,1) + 1*Q(5,1) = 3*14 + 2*30 + 55
/// assert_eq!(iterated_segment_iterated_square_sum(3, 3, 1, 2), 157);
/// assert_eq!(iterated_segment_iterated_square_sum(1, 6, 1, 2), iterated_square_sum(6, 3));
/// ```
pub fn iterated_segment_iterated_square_sum(
    start: u64,
    count: u64,
    order: u64,
    depth: u64,
) -> u128 {
    let overflow = "iterated_segment_iterated_square_sum: result exceeds u128";
    if count == 0 {
        return 0;
    }
    if depth == 0 {
        let last = start.checked_add(count - 1).expect(overflow);
        return iterated_square_sum(last, order);
    }
    if depth == 1 && order == 0 {
        return rectangular_frustum_sum(start, start, count);
    }
    if start == 0 {
        // iterated_square_sum(0, order) == 0, so drop the leading term; the
        // remaining weights are exactly those of a segment one shorter.
        return iterated_segment_iterated_square_sum(1, count - 1, order, depth);
    }

    let p = start - 1;
    let top = order.checked_add(depth).expect(overflow);
    let end = p.checked_add(count).expect(overflow);

    let mut subtrahend = 0u128;
    for t in 0..depth {
        let offset_term = iterated_square_sum(p, top - t);
        if offset_term == 0 {
            continue;
        }
        let product = multiset(count as u128, t as u128)
            .checked_mul(offset_term)
            .expect(overflow);
        subtrahend = subtrahend.checked_add(product).expect(overflow);
    }
    // The minuend is the full iterated sum up to `end`; the subtracted
    // terms are its contributions from indices up to `p`, so no underflow.
    iterated_square_sum(end, top) - subtrahend
}

/// [`iterated_segment_iterated_square_sum`] expanded in the offset
/// `p = start - 1`.
///
/// Writing `S(n, q) = simplex_number(n, q)`, `Q(n, q) =
/// iterated_square_sum(n, q)` and `n = count`, substitute
/// `(p + i)^2 = p^2 + 2p * i + i^2` in the part of every summand beyond
/// index `p`, and split the part up to `p` with the Chu–Vandermonde
/// convolution:
///
/// ```text
/// sum_{i=0}^{n-1} M(n - i, depth - 1) * Q(p + 1 + i, order)
///     == Q(n, order + depth) + 2p * S(n, order + depth)
///      + sum_{j=0}^{order} Q(p, j) * S(n, order + depth - 1 - j)
/// ```
///
/// Every term is a value for the length alone times a value for the offset
/// alone. For `order = 1, depth = 2` (linear weights on a segment of sums of
/// squares) this is `Q(n, 3) + 2p * S(n, 3) + p^2 * S(n, 2) + Q(p, 1) * S(n, 1)`.
///
/// `depth = 0` returns the last term of the segment (or `0` when empty).
/// Runs in `O(order * (order + depth))` time.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::{
///     iterated_segment_iterated_square_sum,
///     iterated_segment_iterated_square_sum_offset_expansion,
/// };
///
/// // Q(3,3) + 4 S(3,3) + 4 S(3,2) + 5 S(3,1) = 27 + 60 + 40 + 30
/// assert_eq!(iterated_segment_iterated_square_sum_offset_expansion(3, 3, 1, 2), 157);
/// assert_eq!(
///     iterated_segment_iterated_square_sum_offset_expansion(4, 6, 2, 3),
///     iterated_segment_iterated_square_sum(4, 6, 2, 3),
/// );
/// ```
pub fn iterated_segment_iterated_square_sum_offset_expansion(
    start: u64,
    count: u64,
    order: u64,
    depth: u64,
) -> u128 {
    let overflow = "iterated_segment_iterated_square_sum_offset_expansion: result exceeds u128";
    if count == 0 {
        return 0;
    }
    if depth == 0 {
        let last = start.checked_add(count - 1).expect(overflow);
        return iterated_square_sum(last, order);
    }
    if start == 0 {
        // iterated_square_sum(0, order) == 0, so drop the leading term.
        return iterated_segment_iterated_square_sum_offset_expansion(1, count - 1, order, depth);
    }

    let p = start - 1;
    let n = count as u128;
    let top = order.checked_add(depth).expect(overflow);

    let linear = simplex(n, top as u128)
        .checked_mul(2 * p as u128)
        .expect(overflow);
    let mut total = iterated_square_sum(count, top)
        .checked_add(linear)
        .expect(overflow);
    for j in 0..=order {
        let offset_term = iterated_square_sum(p, j);
        if offset_term == 0 {
            continue;
        }
        let product = offset_term
            .checked_mul(simplex(n, (top - 1 - j) as u128))
            .expect(overflow);
        total = total.checked_add(product).expect(overflow);
    }
    total
}

/// `depth`-fold iterated partial sum of a segment of squares, i.e. the
/// weighted sum (weights are multiset coefficients `M(x, p) = C(x + p - 1, p)`)
///
/// ```text
/// sum_{i=0}^{count-1} M(count - i, depth - 1) * (start + i)^2
/// ```
///
/// `depth = 1` is the plain segment sum `start^2 + ... + (start + count - 1)^2`
/// and `depth = 2` weights the terms `count, count - 1, ..., 1`, which equals
/// `sum_{k=1}^{count} (start^2 + ... + (start + k - 1)^2)`.
///
/// Evaluated by Newton's forward-difference expansion: `f(x) = x^2` has
/// forward differences `start^2`, `2 * start + 1`, `2` at `start` and none
/// beyond, so
///
/// ```text
/// start^2 * C(count + depth - 1, depth)
///     + (2 * start + 1) * C(count + depth - 1, depth + 1)
///     + 2 * C(count + depth - 1, depth + 2)
/// ```
///
/// (this is [`iterated_segment_power_sum`] with `power = 2`; see
/// [`newton_iterated_sum`](crate::interpolation::newton_iterated_sum) for
/// arbitrary difference tables). `depth = 0` returns the last term
/// `(start + count - 1)^2`, or `0` for an empty segment.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::iterated_segment_square_sum;
///
/// assert_eq!(iterated_segment_square_sum(1, 5, 1), 55);
/// assert_eq!(iterated_segment_square_sum(3, 3, 1), 50); // 9 + 16 + 25
/// assert_eq!(iterated_segment_square_sum(3, 3, 2), 84); // 3*9 + 2*16 + 1*25
/// ```
pub fn iterated_segment_square_sum(start: u64, count: u64, depth: u64) -> u128 {
    iterated_segment_power_sum(start, count, 2, depth)
}

/// `depth`-fold iterated partial sum of a segment of `power`-th powers:
///
/// ```text
/// sum_{i=0}^{count-1} M(count - i, depth - 1) * (start + i)^power
/// ```
///
/// with multiset-coefficient weights `M(x, t) = C(x + t - 1, t)`, so
/// `depth = 1` is the plain segment sum `start^p + ... + (start + count - 1)^p`
/// and `depth = 2` weights the terms `count, count - 1, ..., 1`.
///
/// Evaluated by Newton's forward-difference expansion with the exact
/// differences of `x^p` at `start` (see
/// [`power_forward_differences`](crate::interpolation::power_forward_differences)):
///
/// ```text
/// sum_{j=0}^{p} C(count + depth - 1, depth + j) * Δ^j start^p
/// ```
///
/// For cubes and `depth = 1` this is
/// `m^3 n + (3m^2 + 3m + 1) C(n, 2) + (6m + 6) C(n, 3) + 6 C(n, 4)`; with
/// `depth = 2` every binomial moves up one row:
/// `m^3 C(n+1, 2) + (3m^2 + 3m + 1) C(n+1, 3) + (6m + 6) C(n+1, 4) + 6 C(n+1, 5)`.
///
/// `depth = 0` returns the last term `(start + count - 1)^p`, or `0` for an
/// empty segment. `0^0` is taken to be `1`. Runs in `O(power^2)` time.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::iterated_segment_power_sum;
///
/// assert_eq!(iterated_segment_power_sum(2, 3, 3, 1), 99); // 8 + 27 + 64
/// assert_eq!(iterated_segment_power_sum(2, 3, 3, 2), 142); // 3*8 + 2*27 + 1*64
/// assert_eq!(iterated_segment_power_sum(3, 3, 2, 1), 50); // 9 + 16 + 25
/// ```
pub fn iterated_segment_power_sum(start: u64, count: u64, power: u64, depth: u64) -> u128 {
    if count == 0 {
        return 0;
    }
    // Weights C(count + depth - 1, depth + j) vanish for j >= count, so the
    // differences beyond that order are never needed.
    let differences = power_differences(start as u128, power, count - 1);
    newton_sum(
        &differences,
        count,
        depth,
        "iterated_segment_power_sum: result exceeds u128",
    )
}

/// Power sum iterated `depth` times: `sum_{k=1}^{n} ... sum_{k=1}^{n} k^power`
/// with `depth` summation signs, equivalently
/// `sum_{k=1}^{n} M(n - k + 1, depth - 1) * k^power`.
///
/// Generalizes [`iterated_square_sum`] (`power = 2`) to any power. Each
/// further summation is the same as weighting by `n - k + 1` once more. For
/// cubes:
///
/// ```text
/// iterated_power_sum(n, 3, 1) == T(n)^2                          (T(n) = n(n+1)/2)
/// iterated_power_sum(n, 3, 2) == simplex_number(n, 2) * (3n^2 + 6n + 1) / 10
/// ```
///
/// and in Newton form (differences of `k^3` at `k = 1` are `1, 7, 12, 6`)
/// `iterated_power_sum(n, 3, depth) == sum_j C(n + depth - 1, depth + j) * [1, 7, 12, 6]_j`.
///
/// `depth = 0` returns `n^power` (and `0` for `n = 0`).
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::iterated_power_sum;
///
/// assert_eq!(iterated_power_sum(4, 3, 1), 100); // 1 + 8 + 27 + 64 = 10^2
/// assert_eq!(iterated_power_sum(4, 3, 2), 146); // 1 + 9 + 36 + 100
/// ```
pub fn iterated_power_sum(n: u64, power: u64, depth: u64) -> u128 {
    iterated_segment_power_sum(1, n, power, depth)
}

/// Sum of the first `n` `power`-th powers, `1^p + 2^p + ... + n^p`.
///
/// Computed exactly from the Stirling numbers of the second kind:
/// `sum_{k=1}^{n} k^p == sum_{j} S(p, j) * j! * C(n + 1, j + 1)` (each term is
/// non-negative). `power = 1` gives the triangular numbers, `power = 2` the
/// square pyramidal numbers, and `power = 3` the squared triangular numbers
/// `T(n)^2`. Note `k * k^2` is `k^3`: weighting the squares by their index
/// raises the power by one, so index-weighted sums of powers are again
/// power sums.
///
/// # Panics
///
/// Panics if the result does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::series::power_sum;
///
/// assert_eq!(power_sum(5, 3), 225); // 15^2
/// assert_eq!(power_sum(4, 4), 354); // 1 + 16 + 81 + 256
/// assert_eq!(power_sum(5, 0), 5);
/// ```
pub fn power_sum(n: u64, power: u64) -> u128 {
    iterated_power_sum(n, power, 1)
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

    // C(n + 2, 3), the n-th tetrahedral number.
    let triple = simplex(n, 2);

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
