//! Internal arithmetic helpers shared across modules.

use crate::combinations::checked_combinations;

/// Greatest common divisor of `a` and `b`.
pub(crate) fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

/// Binomial coefficient `C(n, k)` with `u128` arguments.
///
/// Returns `0` when `k > n`. Uses the symmetry `C(n, k) == C(n, n - k)` and
/// an exact division at every step to avoid intermediate overflow.
///
/// Returns `None` if `C(n, k)` does not fit in `u128`.
pub(crate) fn binomial_wide(n: u128, k: u128) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        result = mul_div(result, n - i + 1, i)?;
    }
    Some(result)
}

/// Computes `result * num / den` exactly, cancelling common factors before
/// multiplying so intermediate values stay as small as possible.
///
/// Returns `None` if the result (or the product before the division becomes
/// exact) exceeds `u128`.
///
/// # Panics
///
/// Panics if `den == 0`.
pub(crate) fn mul_div(result: u128, num: u128, den: u128) -> Option<u128> {
    assert!(den > 0, "mul_div: zero denominator");

    let g = gcd(num, den);
    let (num, mut den) = (num / g, den / g);

    let g = gcd(result, den);
    let result = result / g;
    den /= g;

    if result.is_multiple_of(den) {
        (result / den).checked_mul(num)
    } else if num.is_multiple_of(den) {
        (num / den).checked_mul(result)
    } else {
        Some(result.checked_mul(num)? / den)
    }
}

/// Splits the product `x * y` by `h` without forming the product:
/// returns `(q, r)` with `x * y == q * h + r` and `r < h`.
///
/// Intended for small `h`, where `(x % h) * y` cannot overflow.
///
/// Returns `None` if the quotient exceeds `u128`.
///
/// # Panics
///
/// Panics if `h == 0`.
pub(crate) fn mul_div_rem(x: u128, y: u128, h: u128) -> Option<(u128, u128)> {
    let (xq, xr) = (x / h, x % h);
    let low = xr.checked_mul(y)?;
    let q = xq.checked_mul(y)?.checked_add(low / h)?;
    Some((q, low % h))
}

/// The `n`-th simplex number of the given dimension,
/// `C(n + dimension, dimension + 1)`.
///
/// Built from `n` (dimension 0) by the ratio recurrence between successive
/// dimensions,
///
/// ```text
/// simplex(n, d + 1) == simplex(n, d) * (n + d + 1) / (d + 2)
/// ```
///
/// with an exact division at every step.
///
/// Returns `None` if the value exceeds `u128`.
pub(crate) fn simplex(n: u128, dimension: u128) -> Option<u128> {
    let mut result = n;
    for d in 0..dimension {
        result = mul_div(result, n + d + 1, d + 2)?;
    }
    Some(result)
}

/// Multiset coefficient `M(n, k) = C(n + k - 1, k)`, with `M(n, 0) == 1` and
/// `M(0, k) == 0` for `k > 0`. For `k >= 1`, `M(n, k) == simplex(n, k - 1)`.
///
/// Returns `None` if the value exceeds `u128`.
pub(crate) fn multiset(n: u128, k: u128) -> Option<u128> {
    if k == 0 {
        return Some(1);
    }
    if n == 0 {
        return Some(0);
    }

    // C(n + k - 1, k) == C(n + k - 1, n - 1); iterate over the smaller one.
    let top = n + k - 1;
    let k = k.min(n - 1);

    let mut result = 1u128;
    for i in 1..=k {
        result = mul_div(result, top - k + i, i)?;
    }
    Some(result)
}

/// Row `[S(n, 0), S(n, 1), ..., S(n, n)]` of Stirling numbers of the second
/// kind, from `S(i, k) = k * S(i - 1, k) + S(i - 1, k - 1)`.
///
/// Returns `None` if an entry exceeds `u128`.
pub(crate) fn stirling2_row(n: u64) -> Option<Vec<u128>> {
    let mut row = vec![1u128];
    for i in 1..=n as usize {
        let mut next = vec![0u128; i + 1];
        for k in 1..=i {
            let carried = match row.get(k) {
                Some(&s) => (k as u128).checked_mul(s)?,
                None => 0,
            };
            next[k] = carried.checked_add(row[k - 1])?;
        }
        row = next;
    }
    Some(row)
}

/// Falling factorial `x (x - 1) ... (x - t + 1)`; `0` when `t > x`.
///
/// Returns `None` if the value exceeds `u128`.
fn falling(x: u128, t: u128) -> Option<u128> {
    if t > x {
        return Some(0);
    }
    (0..t).try_fold(1u128, |acc, i| acc.checked_mul(x - i))
}

/// Forward differences `[Δ^0 f(m), ..., Δ^{max_order} f(m)]` of
/// `f(x) = x^power` at `x = m`, truncated at `min(power, max_order)` (all
/// higher differences vanish).
///
/// Expanding `x^power = sum_k S(power, k) x^(k)` in falling factorials and
/// using `Δ^j x^(k) = k^(j) x^(k - j)`:
///
/// ```text
/// Δ^j m^power == sum_{k=j}^{power} S(power, k) * k^(j) * m^(k - j)
/// ```
///
/// Every term is non-negative, so each difference is computed exactly.
/// `0^0` is taken to be `1`.
///
/// Returns `None` if a difference exceeds `u128`.
pub(crate) fn power_differences(m: u128, power: u64, max_order: u64) -> Option<Vec<u128>> {
    let stirling = stirling2_row(power)?;
    (0..=power.min(max_order) as u128)
        .map(|j| {
            (j..=power as u128)
                .filter(|&k| stirling[k as usize] != 0)
                .try_fold(0u128, |acc, k| {
                    let tail = falling(m, k - j)?;
                    if tail == 0 {
                        return Some(acc);
                    }
                    let term = stirling[k as usize]
                        .checked_mul(falling(k, j)?)?
                        .checked_mul(tail)?;
                    acc.checked_add(term)
                })
        })
        .collect()
}

/// `depth`-fold iterated partial sum of the first `count` terms of a
/// sequence from its non-negative leading forward differences
/// `[Δ^0 f(0), Δ^1 f(0), ...]`:
/// `sum_j C(count + depth - 1, depth + j) * Δ^j f(0)`.
///
/// Returns `None` if the result exceeds `u128`.
pub(crate) fn newton_sum(differences: &[u128], count: u64, depth: u64) -> Option<u128> {
    if count == 0 {
        return Some(0);
    }
    // count >= 1, so this cannot underflow.
    let top = (count - 1).checked_add(depth)?;

    let mut total = 0u128;
    for (j, &delta) in (0u64..).zip(differences) {
        if delta == 0 {
            continue;
        }
        let Some(order) = depth.checked_add(j).filter(|&order| order <= top) else {
            break;
        };
        let term = checked_combinations(top, order)?.checked_mul(delta)?;
        total = total.checked_add(term)?;
    }
    Some(total)
}
