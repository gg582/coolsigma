//! Exact positive integer roots of polynomials with integer coefficients.

/// Positive integer root of a polynomial with integer coefficients, or
/// `None` if there is none.
///
/// `coefficients[i]` is the coefficient of `x^i` (constant term first).
/// Leading and trailing zero coefficients are ignored.
///
/// Two-stage hybrid:
///
/// 1. When the constant term is small (`|c0| <= 10^12`), the candidates are
///    the positive divisors of `|c0|` (rational-root theorem) tested in
///    ascending order with exact `i128` Horner evaluation, so the *smallest*
///    positive integer root is returned.
/// 2. For larger constant terms, divisor enumeration (linear in
///    `sqrt(|c0|)`) is infeasible, so a Newton–Raphson search in `f64`
///    starting from the Cauchy bound is tried first and its result verified
///    with an exact integer evaluation. This stage returns *any* verified
///    positive integer root, not necessarily the smallest.
///
/// Exact evaluation is exact `i128` Horner. Once a Horner partial sum
/// overflows `i128`, the remaining multiplications cannot shrink it below
/// the overflow point again, so the candidate is certainly not a root and
/// is skipped. (Formally this relies on coefficients being strictly smaller
/// than `2^127` in magnitude, which holds for every `i128` input except the
/// pathological `i128::MIN`; coefficients that large are outside the
/// supported domain.)
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

    if c0 <= ENUMERATION_LIMIT {
        smallest_root_by_divisors(coeffs, c0)
    } else {
        newton_root(coeffs)
    }
}

/// Divisor enumeration is linear in `sqrt(c0)`; past this bound the
/// Newton–Raphson stage is used instead.
const ENUMERATION_LIMIT: u128 = 1_000_000_000_000;

/// Exact ascending divisor enumeration: returns the smallest positive
/// integer root.
fn smallest_root_by_divisors(coeffs: &[i128], c0: u128) -> Option<u64> {
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

/// Newton–Raphson search in `f64` starting from the Cauchy bound
/// `1 + max|a_i| / |a_n|`, followed by exact integer verification of the
/// converged candidate (and its neighbours, to absorb rounding).
fn newton_root(coeffs: &[i128]) -> Option<u64> {
    let lead = coeffs.last().unwrap().unsigned_abs() as f64;
    let max_coeff = coeffs[..coeffs.len() - 1]
        .iter()
        .map(|c| c.unsigned_abs())
        .max()
        .unwrap_or(0) as f64;
    let mut x = 1.0 + max_coeff / lead;

    for _ in 0..100 {
        let (f, df) = eval_with_derivative_f64(coeffs, x);
        if !f.is_finite() || !df.is_finite() || df == 0.0 {
            return None;
        }
        let next = x - f / df;
        if !next.is_finite() || next < 0.5 {
            return None;
        }
        if (next - x).abs() <= 1e-12 * x {
            x = next;
            break;
        }
        x = next;
    }

    let center = x.round();
    if center < 1.0 || center > u64::MAX as f64 {
        return None;
    }
    let center = center as i128;

    [center - 1, center, center + 1]
        .into_iter()
        .find(|&r| r >= 1 && eval(coeffs, r) == Some(0))
        .and_then(|r| u64::try_from(r).ok())
}

/// Simultaneous Horner evaluation of `f(x)` and `f'(x)` in `f64`.
fn eval_with_derivative_f64(coeffs: &[i128], x: f64) -> (f64, f64) {
    let mut f = 0.0;
    let mut df = 0.0;
    for &c in coeffs.iter().rev() {
        df = df * x + f;
        f = f * x + c as f64;
    }
    (f, df)
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
