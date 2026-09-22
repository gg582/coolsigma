//! Newton forward-difference interpolation.

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
