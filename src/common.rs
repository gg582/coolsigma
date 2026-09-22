//! Internal arithmetic helpers shared across modules.

/// Greatest common divisor of `a` and `b`.
pub(crate) fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

/// Computes `result * num / den` exactly, cancelling common factors before
/// multiplying so intermediate values stay as small as possible.
///
/// Callers must ensure the mathematical result fits in `u128`.
///
/// # Panics
///
/// Panics if `den == 0` or the result exceeds `u128`.
pub(crate) fn mul_div(result: u128, num: u128, den: u128) -> u128 {
    assert!(den > 0, "mul_div: zero denominator");

    let g = gcd(num, den);
    let (num, mut den) = (num / g, den / g);

    let g = gcd(result, den);
    let result = result / g;
    den /= g;

    if result.is_multiple_of(den) {
        result / den * num
    } else if num.is_multiple_of(den) {
        num / den * result
    } else {
        let product = result
            .checked_mul(num)
            .expect("mul_div: result exceeds u128");
        product / den
    }
}
