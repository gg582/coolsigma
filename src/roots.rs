//! Integer root finding for polynomial equations (the "opening the square"
//! step of the classical problems).

/// Smallest positive integer root of a polynomial with integer coefficients,
/// or `None` if there is none.
///
/// `coefficients[i]` is the coefficient of `x^i` (constant term first).
/// Leading and trailing zero coefficients are ignored. Candidates are the
/// positive divisors of the constant term (rational-root theorem), tested
/// ascending, so the *smallest* positive integer root is returned.
///
/// Evaluation is exact (`i128` Horner). Once a Horner partial sum overflows
/// `i128`, the remaining multiplications cannot shrink it below the overflow
/// point again, so the candidate is certainly not a root and is skipped.
///
/// Runs in `O(sqrt(|c0|) * degree)` time, where `c0` is the constant term.
///
/// # Panics
///
/// Panics if `coefficients` is empty or consists only of zeros.
///
/// # Example
///
/// ```
/// use coolsigma::roots::positive_integer_root;
///
/// // n^3 + 30 n^2 + 29 n - 6396 == 0
/// assert_eq!(positive_integer_root(&[-6396, 29, 30, 1]), Some(12));
/// // x^2 + 1 == 0 has no integer root.
/// assert_eq!(positive_integer_root(&[1, 0, 1]), None);
/// ```
pub fn positive_integer_root(coefficients: &[i128]) -> Option<u64> {
    // Trim zero coefficients on both ends (leading zeros change nothing;
    // trailing zeros factor out x, which has no positive root contribution).
    let mut coeffs = coefficients;
    while let Some((&0, rest)) = coeffs.split_last() {
        coeffs = rest;
    }
    while let Some((&0, rest)) = coeffs.split_first() {
        coeffs = rest;
    }
    assert!(!coeffs.is_empty(), "polynomial has no nonzero coefficients");

    if coeffs.len() == 1 {
        return None; // constant polynomial: no root
    }

    let c0 = coeffs[0].unsigned_abs();

    // Candidate roots are the positive divisors of |c0|, ascending.
    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut d = 1u128;
    while d * d <= c0 {
        if c0.is_multiple_of(d) {
            small.push(d);
            if d * d != c0 {
                large.push(c0 / d);
            }
        }
        d += 1;
    }

    for candidate in small.into_iter().chain(large.into_iter().rev()) {
        if eval(coeffs, candidate as i128) == Some(0) {
            return u64::try_from(candidate).ok();
        }
    }
    None
}

/// Exact Horner evaluation, or `None` once the running value overflows
/// `i128` (in which case the final value is certainly nonzero: each
/// remaining step multiplies by `x >= 1` before adding a coefficient).
fn eval(coefficients: &[i128], x: i128) -> Option<i128> {
    let mut acc = 0i128;
    for &c in coefficients.iter().rev() {
        acc = acc.checked_mul(x)?.checked_add(c)?;
    }
    Some(acc)
}
