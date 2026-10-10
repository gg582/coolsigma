//! Newton forward-difference interpolation.

use crate::combinations::combinations;
use crate::common::power_differences;

/// First forward difference of two consecutive terms of a sequence:
/// `current - previous`.
///
/// Requires `current >= previous`, which holds for any non-decreasing
/// integer sequence.
///
/// # Example
///
/// ```
/// use coolsigma::interpolation::forward_difference;
///
/// assert_eq!(forward_difference(3, 10), 7);
/// ```
pub fn forward_difference(previous: u128, current: u128) -> u128 {
    current - previous
}

/// Evaluates `f(w + s)` from `f(w)` and the forward differences at `w`:
///
/// ```text
/// f(w) + s*d1 + s(s-1)/2! * d2 + s(s-1)(s-2)/3! * d3 + ...
/// ```
///
/// `differences` holds `[d1, d2, d3, ...]`; any number of difference orders
/// is accepted and all are used.
///
/// # Example
///
/// ```
/// use coolsigma::interpolation::newton_forward_interpolation;
///
/// let diffs = [2.0, 0.5, 0.1];
/// let value = newton_forward_interpolation(10.0, 0.5, &diffs);
/// assert!((value - 10.94375).abs() < 1e-12);
/// ```
pub fn newton_forward_interpolation(base: f64, offset: f64, differences: &[f64]) -> f64 {
    let mut result = base;
    let mut falling = 1.0; // s^{underline r}, the falling factorial
    let mut factorial = 1.0;

    for (i, &delta) in differences.iter().enumerate() {
        let r = i as f64 + 1.0;
        falling *= offset - i as f64;
        factorial *= r;
        result += falling / factorial * delta;
    }

    result
}

/// Leading forward differences `[f(0), Δf(0), Δ²f(0), ...]` of a sequence
/// given by its values `[f(0), f(1), ..., f(len - 1)]`.
///
/// For a polynomial of degree `d` sampled at `d + 1` or more points, the
/// differences past order `d` are zero.
///
/// # Panics
///
/// Panics if a difference does not fit in `i128`.
///
/// # Example
///
/// ```
/// use coolsigma::interpolation::leading_differences;
///
/// // k^2 at k = 3, 4, 5, 6: 3^2, 2*3 + 1, 2, 0
/// assert_eq!(leading_differences(&[9, 16, 25, 36]), vec![9, 7, 2, 0]);
/// ```
pub fn leading_differences(values: &[i128]) -> Vec<i128> {
    let mut row = values.to_vec();
    let mut leading = Vec::with_capacity(values.len());
    while let Some(&first) = row.first() {
        leading.push(first);
        row = row
            .windows(2)
            .map(|w| {
                w[1].checked_sub(w[0])
                    .expect("leading_differences: difference exceeds i128")
            })
            .collect();
    }
    leading
}

/// `depth`-fold iterated partial sum of the first `count` terms of a
/// sequence, from its value `base = f(0)` and its forward differences
/// `differences = [Δf(0), Δ²f(0), ...]` at the first term:
///
/// ```text
/// sum_{j>=0} C(count + depth - 1, j + depth) * Δ^j f(0)
/// ```
///
/// `depth = 1` is the plain sum `f(0) + ... + f(count - 1)`, `depth = 2` the
/// weighted sum `sum_{i=0}^{count-1} (count - i) * f(i)` (the sum of the
/// running totals), and in general the weight of `f(i)` is the multiset
/// coefficient `C(count - i + depth - 2, depth - 1)`. `depth = 0` reduces to
/// Newton's forward formula and returns `f(count - 1)` (or `0` when
/// `count == 0`).
///
/// The result is exact when the difference table is complete, i.e. when `f`
/// is a polynomial whose degree is less than `1 + differences.len()`.
///
/// Integer-valued but signed, since differences of a decreasing sequence are
/// negative.
///
/// # Panics
///
/// Panics if the result or an intermediate term does not fit in `i128`.
///
/// # Example
///
/// ```
/// use coolsigma::interpolation::newton_iterated_sum;
///
/// // 1^2 + ... + 5^2 = 5 + 3 C(5,2) + 2 C(5,3)
/// assert_eq!(newton_iterated_sum(1, &[3, 2], 5, 1), 55);
/// // 3*3^2 + 2*4^2 + 1*5^2
/// assert_eq!(newton_iterated_sum(9, &[7, 2], 3, 2), 84);
/// // 10 + 7 + 4 + 1 (decreasing, so the difference is negative)
/// assert_eq!(newton_iterated_sum(10, &[-3], 4, 1), 22);
/// ```
pub fn newton_iterated_sum(base: i128, differences: &[i128], count: u64, depth: u64) -> i128 {
    if count == 0 {
        return 0;
    }

    let overflow = "newton_iterated_sum: result exceeds i128";
    let top = (count - 1).checked_add(depth).expect(overflow);

    let mut total = 0i128;
    for (j, &delta) in (0u64..).zip(std::iter::once(&base).chain(differences)) {
        if delta == 0 {
            continue;
        }
        let Some(order) = depth.checked_add(j).filter(|&order| order <= top) else {
            break;
        };
        let weight = i128::try_from(combinations(top, order)).expect(overflow);
        let term = weight.checked_mul(delta).expect(overflow);
        total = total.checked_add(term).expect(overflow);
    }
    total
}

/// Leading forward differences `[m^p, Δ m^p, Δ² m^p, ..., Δ^p m^p]` of the
/// power `f(x) = x^p` at `x = start`, with `p = power`.
///
/// Through the Stirling numbers of the second kind `S(p, k)`,
///
/// ```text
/// Δ^j m^p == sum_{k=j}^{p} S(p, k) * k! / (k - j)! * m! / (m - k + j)!
/// ```
///
/// so every difference is non-negative and exact. For cubes this gives
/// `m^3`, `3m^2 + 3m + 1`, `6m + 6`, `6`. Differences past order `p` are
/// zero and omitted. `0^0` is taken to be `1`.
///
/// Feed the result to [`newton_iterated_sum`] (after converting to `i128`)
/// or use [`iterated_segment_power_sum`](crate::series::iterated_segment_power_sum)
/// directly.
///
/// # Panics
///
/// Panics if a difference does not fit in `u128`.
///
/// # Example
///
/// ```
/// use coolsigma::interpolation::power_forward_differences;
///
/// assert_eq!(power_forward_differences(3, 2), vec![9, 7, 2]);
/// assert_eq!(power_forward_differences(3, 3), vec![27, 37, 24, 6]);
/// ```
pub fn power_forward_differences(start: u64, power: u64) -> Vec<u128> {
    power_differences(start as u128, power, power)
}
