//! Cross-checks of the closed-form implementations against independent
//! brute-force reference implementations (Pascal's triangle and direct
//! summation), run over grids of small inputs.

use coolsigma::combinations::combinations;
use coolsigma::series::{
    arithmetic_series_sum, centered_expansion, generalized_series_sum, interleaved_series_sum,
    iterated_square_sum, weighted_partial_square_sum,
};
use coolsigma::sums::{
    segment_simplex_sum, simplex_number, simplex_sum, truncated_simplex_sum,
};

/// Binomial coefficient `C(n, k)` computed additively via Pascal's triangle.
/// Independent of the multiplicative implementations under test.
fn pascal(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k) as usize;
    let mut row: Vec<u128> = vec![1];
    for _ in 0..n {
        let mut next = vec![1u128; row.len() + 1];
        for j in 1..row.len() {
            next[j] = row[j - 1] + row[j];
        }
        row = next;
    }
    row[k]
}

#[test]
fn simplex_sum_matches_pascal_summation() {
    for d in 0..=8u64 {
        for m in 0..=40u64 {
            let expected: u128 = (1..=m).map(|i| pascal(i + d - 1, d)).sum();
            assert_eq!(simplex_sum(d, m), expected, "simplex_sum({d}, {m})");
        }
    }
}

#[test]
fn simplex_sum_matches_binomial_identity() {
    for d in 0..=8u64 {
        for m in 0..=40u64 {
            assert_eq!(
                simplex_sum(d, m),
                combinations(d + m, d + 1),
                "simplex_sum({d}, {m})"
            );
        }
    }
}

#[test]
fn truncated_simplex_sum_matches_pascal_summation() {
    for d in 0..=8u64 {
        for m in 1..=40u64 {
            for start in 1..=m {
                let expected: u128 = (start..=m).map(|i| pascal(i + d - 1, d)).sum();
                assert_eq!(
                    truncated_simplex_sum(start, d, m),
                    expected,
                    "truncated_simplex_sum({start}, {d}, {m})"
                );
            }
        }
    }
}

#[test]
fn arithmetic_series_sum_matches_naive_loop() {
    for first in 0..=20u64 {
        for count in 0..=30u64 {
            for difference in 0..=10u64 {
                let expected: u128 = (0..count)
                    .map(|i| first as u128 + i as u128 * difference as u128)
                    .sum();
                assert_eq!(
                    arithmetic_series_sum(first, count, difference),
                    expected,
                    "arithmetic_series_sum({first}, {count}, {difference})"
                );
            }
        }
    }
}

#[test]
fn centered_expansion_matches_naive_loop() {
    for sides in 0..=12u64 {
        for layer in 0..=40u64 {
            let expected = 1 + sides as u128 * (1..=layer).map(|k| k as u128).sum::<u128>();
            assert_eq!(
                centered_expansion(sides, layer),
                expected,
                "centered_expansion({sides}, {layer})"
            );
        }
    }
}

#[test]
fn simplex_number_matches_pascal() {
    for d in 0..=6u64 {
        for n in 0..=40u64 {
            assert_eq!(
                simplex_number(n, d),
                pascal(n + d, d + 1),
                "simplex_number({n}, {d})"
            );
        }
    }
}

#[test]
fn segment_simplex_sum_matches_naive_loop() {
    for d in 0..=5u64 {
        for start in 0..=10u64 {
            for count in 0..=15u64 {
                let expected: u128 = (start..start + count)
                    .map(|k| pascal(k + d, d + 1))
                    .sum();
                assert_eq!(
                    segment_simplex_sum(start, count, d),
                    expected,
                    "segment_simplex_sum({start}, {count}, {d})"
                );
            }
        }
    }
}

#[test]
fn generalized_series_sum_matches_pascal_summation() {
    for r in 0..=6u64 {
        for n in 0..=40u64 {
            // Summand identity: f(n) - f(n - 1) = k * C(k + r - 1, r).
            let expected: u128 = (1..=n).map(|k| k as u128 * pascal(k + r - 1, r)).sum();
            assert_eq!(
                generalized_series_sum(n, r),
                expected,
                "generalized_series_sum({n}, {r})"
            );
        }
    }
}

#[test]
fn iterated_square_sum_matches_naive_summation() {
    for n in 0..=30u64 {
        for depth in 1..=5u64 {
            let mut acc: Vec<u128> = (1..=n).map(|k| (k as u128) * (k as u128)).collect();
            for _ in 1..depth {
                for i in 1..acc.len() {
                    acc[i] += acc[i - 1];
                }
            }
            let expected: u128 = acc.iter().sum();
            assert_eq!(
                iterated_square_sum(n, depth),
                expected,
                "iterated_square_sum({n}, {depth})"
            );
        }
    }
}

#[test]
fn weighted_partial_square_sum_matches_naive_summation() {
    for n in 0..=60u64 {
        let mut partial = 0u128;
        let mut expected = 0u128;
        for k in 1..=n {
            partial += (k as u128) * (k as u128);
            expected += k as u128 * partial;
        }
        assert_eq!(
            weighted_partial_square_sum(n),
            expected,
            "weighted_partial_square_sum({n})"
        );
    }
}

#[test]
fn interleaved_series_sum_matches_naive_loop() {
    for n in 1..=100u64 {
        let mut expected = 0u128;
        for k in 1..=n {
            if k % 2 != 0 {
                let m = k.div_ceil(2) as u128;
                expected += 3 * m * m - 3 * m + 1;
            } else {
                let m = (k / 2) as u128;
                expected += 3 * m * m;
            }
        }
        assert_eq!(
            interleaved_series_sum(n),
            expected,
            "interleaved_series_sum({n})"
        );
    }
}
