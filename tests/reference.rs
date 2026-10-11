//! Cross-checks of the closed-form implementations against independent
//! brute-force reference implementations (Pascal's triangle and direct
//! summation), run over grids of small inputs.

use coolsigma::combinations::{combinations, multiset_coefficient, stirling_second_kind};
use coolsigma::interpolation::{
    leading_differences, newton_iterated_sum, power_forward_differences,
};
use coolsigma::series::{
    arithmetic_series_sum, centered_expansion, centered_hexagonal_number,
    centered_triangular_number, generalized_series_sum, interleaved_series_sum,
    iterated_interleaved_series_sum, iterated_power_sum, iterated_segment_iterated_square_sum,
    iterated_segment_iterated_square_sum_offset_expansion, iterated_segment_power_sum,
    iterated_segment_square_sum, iterated_square_sum, odd_square, power_sum,
    rectangular_frustum_sum, segment_generalized_series_sum, segment_iterated_square_sum,
    segment_iterated_square_sum_offset_expansion, weighted_partial_square_sum,
    weighted_shifted_pyramidal_sum, weighted_shifted_square_sum,
};
use coolsigma::sums::{
    ascending_weighted_segment_simplex_sum, iterated_segment_simplex_sum, segment_simplex_sum,
    segment_simplex_sum_vandermonde, simplex_convolution, simplex_number,
    trapezoid_weighted_segment_simplex_sum, weighted_segment_simplex_sum,
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
fn simplex_number_matches_binomial_identity() {
    for d in 0..=8u64 {
        for m in 0..=40u64 {
            assert_eq!(
                simplex_number(m, d),
                combinations(m + d, d + 1),
                "simplex_number({m}, {d})"
            );
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
                centered_expansion(layer, sides),
                expected,
                "centered_expansion({layer}, {sides})"
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
                let expected: u128 = (start..start + count).map(|k| pascal(k + d, d + 1)).sum();
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
fn multiset_coefficient_matches_pascal() {
    for n in 0..=30u64 {
        for k in 0..=12u64 {
            let expected = if n == 0 {
                (k == 0) as u128
            } else {
                pascal(n + k - 1, k)
            };
            assert_eq!(
                multiset_coefficient(n, k),
                expected,
                "multiset_coefficient({n}, {k})"
            );
        }
    }
}

#[test]
fn segment_simplex_sum_vandermonde_matches_naive_loop() {
    for d in 0..=6u64 {
        for start in 0..=12u64 {
            for count in 0..=15u64 {
                let expected: u128 = (start..start + count).map(|k| pascal(k + d, d + 1)).sum();
                assert_eq!(
                    segment_simplex_sum_vandermonde(start, count, d),
                    expected,
                    "segment_simplex_sum_vandermonde({start}, {count}, {d})"
                );
            }
        }
    }
}

#[test]
fn weighted_segment_simplex_sum_matches_naive_loop() {
    for d in 0..=5u64 {
        for start in 0..=10u64 {
            for count in 0..=15u64 {
                let expected: u128 = (0..count)
                    .map(|i| (count - i) as u128 * pascal(start + i + d, d + 1))
                    .sum();
                assert_eq!(
                    weighted_segment_simplex_sum(start, count, d),
                    expected,
                    "weighted_segment_simplex_sum({start}, {count}, {d})"
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
fn weighted_shifted_square_sum_matches_naive_loop() {
    for m in 0..=10u64 {
        for n in 0..=20u64 {
            let mut expected = 0u128;
            for k in 1..=n {
                let partial: u128 = (1..=k)
                    .map(|j| {
                        let x = (m + j - 1) as u128;
                        x * x
                    })
                    .sum();
                expected += k as u128 * partial;
            }
            assert_eq!(
                weighted_shifted_square_sum(m, n),
                expected,
                "weighted_shifted_square_sum({m}, {n})"
            );
        }
    }
}

#[test]
fn weighted_shifted_pyramidal_sum_matches_naive_loop() {
    for m in 0..=10u64 {
        for n in 0..=20u64 {
            let mut expected = 0u128;
            for k in 1..=n {
                let pyramidal: u128 = (1..=m + k - 1)
                    .map(|i| (i as u128) * (i as u128))
                    .sum();
                expected += k as u128 * pyramidal;
            }
            assert_eq!(
                weighted_shifted_pyramidal_sum(m, n),
                expected,
                "weighted_shifted_pyramidal_sum({m}, {n})"
            );
        }
    }
}

#[test]
fn centered_numbers_match_naive_loop() {
    for n in 0..=40u64 {
        let triangular: u128 = (1..=n).map(|k| k as u128).sum();
        assert_eq!(centered_triangular_number(n), 1 + 3 * triangular);
        assert_eq!(centered_hexagonal_number(n), 1 + 6 * triangular);
        assert_eq!(odd_square(n), 1 + 8 * triangular);
    }
}

#[test]
fn interleaved_series_sum_matches_cone_pile_formula() {
    for n in 1..=100u64 {
        // The reference summary's closed form:
        // W_n == (S_n^1 * (2n + 1) + ceil(n / 2)) / 4.
        let s1 = simplex_number(n, 1);
        let expected = (s1 * (2 * n as u128 + 1) + n.div_ceil(2) as u128) / 4;
        assert_eq!(interleaved_series_sum(n), expected, "interleaved_series_sum({n})");
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

/// Applies `depth` running-total passes to `terms` and returns the last
/// entry: the `depth`-fold iterated partial sum. `depth = 0` returns the
/// last term itself.
fn iterate_partial_sums(mut terms: Vec<u128>, depth: u64) -> u128 {
    for _ in 0..depth {
        for i in 1..terms.len() {
            terms[i] += terms[i - 1];
        }
    }
    terms.last().copied().unwrap_or(0)
}

/// `depth`-fold iterated sum of squares by direct accumulation.
fn naive_iterated_square_sum(n: u64, depth: u64) -> u128 {
    if n == 0 {
        return 0;
    }
    iterate_partial_sums((1..=n).map(|k| (k as u128) * (k as u128)).collect(), depth)
}

#[test]
fn iterated_segment_simplex_sum_matches_naive_loop() {
    for d in 0..=4u64 {
        for depth in 0..=4u64 {
            for start in 0..=8u64 {
                for count in 0..=10u64 {
                    let terms = (start..start + count)
                        .map(|k| pascal(k + d, d + 1))
                        .collect();
                    assert_eq!(
                        iterated_segment_simplex_sum(start, count, d, depth),
                        iterate_partial_sums(terms, depth),
                        "iterated_segment_simplex_sum({start}, {count}, {d}, {depth})"
                    );
                }
            }
        }
    }
}

#[test]
fn simplex_number_is_linearly_weighted_sum_two_dimensions_down() {
    // S(n, r) == sum_{k=1}^{n} (n - k + 1) S(k, r - 2)
    for r in 2..=7u64 {
        for n in 0..=20u64 {
            let expected: u128 = (1..=n)
                .map(|k| (n - k + 1) as u128 * simplex_number(k, r - 2))
                .sum();
            assert_eq!(simplex_number(n, r), expected, "S({n}, {r})");
        }
    }
}

#[test]
fn simplex_number_ratio_recurrence() {
    // S(n, r + 1) * (r + 2) == S(n, r) * (n + r + 1)
    for r in 0..=8u64 {
        for n in 0..=30u64 {
            assert_eq!(
                simplex_number(n, r + 1) * (r as u128 + 2),
                simplex_number(n, r) * (n + r + 1) as u128,
                "S({n}, {r})"
            );
        }
    }
}

#[test]
fn iterated_square_sum_is_rescaled_simplex_number() {
    // Q(n, r) * (r + 2) == S(n, r) * (2n + r)
    for r in 0..=6u64 {
        for n in 0..=30u64 {
            assert_eq!(
                iterated_square_sum(n, r) * (r as u128 + 2),
                simplex_number(n, r) * (2 * n + r) as u128,
                "Q({n}, {r})"
            );
        }
    }
}

#[test]
fn rectangular_frustum_sum_matches_naive_loop() {
    for a in 0..=12u64 {
        for b in 0..=12u64 {
            for layers in 0..=20u64 {
                let expected: u128 = (0..layers).map(|i| (a + i) as u128 * (b + i) as u128).sum();
                assert_eq!(
                    rectangular_frustum_sum(a, b, layers),
                    expected,
                    "rectangular_frustum_sum({a}, {b}, {layers})"
                );
            }
        }
    }
}

#[test]
fn rectangular_frustum_sum_near_u128_limit() {
    // A single huge layer: the formula's bracket alone is 6 * a^2 > u128::MAX.
    let a = 1u64 << 63;
    assert_eq!(rectangular_frustum_sum(a, a, 1), 1u128 << 126);
    let a = u64::MAX;
    assert_eq!(rectangular_frustum_sum(a, a, 1), a as u128 * a as u128);
    // Two layers whose sum fits but whose formula bracket would not.
    let a = 12_000_000_000_000_000_000u64; // result ~2.9e38, bracket ~8.6e38
    let (x, y) = (a as u128, a as u128 + 1);
    assert_eq!(rectangular_frustum_sum(a, a, 2), x * x + y * y);
    assert_eq!(
        rectangular_frustum_sum(a, a + 7, 2),
        x * (x + 7) + y * (y + 7)
    );
}

#[test]
fn segment_iterated_square_sum_matches_naive_loop() {
    for depth in 0..=4u64 {
        for start in 0..=10u64 {
            for count in 0..=12u64 {
                let expected: u128 = (start..start + count)
                    .map(|k| naive_iterated_square_sum(k, depth))
                    .sum();
                assert_eq!(
                    segment_iterated_square_sum(start, count, depth),
                    expected,
                    "segment_iterated_square_sum({start}, {count}, {depth})"
                );
                assert_eq!(
                    segment_iterated_square_sum_offset_expansion(start, count, depth),
                    expected,
                    "segment_iterated_square_sum_offset_expansion({start}, {count}, {depth})"
                );
            }
        }
    }
}

#[test]
fn iterated_segment_square_sum_matches_naive_loop() {
    for depth in 0..=5u64 {
        for start in 0..=10u64 {
            for count in 0..=12u64 {
                let terms = (start..start + count)
                    .map(|k| (k as u128) * (k as u128))
                    .collect();
                assert_eq!(
                    iterated_segment_square_sum(start, count, depth),
                    iterate_partial_sums(terms, depth),
                    "iterated_segment_square_sum({start}, {count}, {depth})"
                );
            }
        }
    }
}

#[test]
fn newton_iterated_sum_matches_naive_loop_for_cubic() {
    // f(x) = 2x^3 - 9x^2 + x - 4 has negative values and differences.
    let f = |x: i128| 2 * x * x * x - 9 * x * x + x - 4;
    for offset in -5..=5i128 {
        let leading = leading_differences(&(0..4).map(|i| f(offset + i)).collect::<Vec<_>>());
        for depth in 0..=4u64 {
            for count in 0..=10u64 {
                let mut terms: Vec<i128> = (0..count as i128).map(|i| f(offset + i)).collect();
                for _ in 0..depth {
                    for i in 1..terms.len() {
                        terms[i] += terms[i - 1];
                    }
                }
                let expected = terms.last().copied().unwrap_or(0);
                assert_eq!(
                    newton_iterated_sum(leading[0], &leading[1..], count, depth),
                    expected,
                    "newton_iterated_sum offset={offset} count={count} depth={depth}"
                );
            }
        }
    }
}

#[test]
fn iterated_segment_iterated_square_sum_matches_naive_loop() {
    for order in 0..=3u64 {
        for depth in 0..=4u64 {
            for start in 0..=8u64 {
                for count in 0..=10u64 {
                    let terms = (start..start + count)
                        .map(|k| naive_iterated_square_sum(k, order))
                        .collect();
                    let expected = iterate_partial_sums(terms, depth);
                    assert_eq!(
                        iterated_segment_iterated_square_sum(start, count, order, depth),
                        expected,
                        "iterated_segment_iterated_square_sum({start}, {count}, {order}, {depth})"
                    );
                    assert_eq!(
                        iterated_segment_iterated_square_sum_offset_expansion(
                            start, count, order, depth
                        ),
                        expected,
                        "..._offset_expansion({start}, {count}, {order}, {depth})"
                    );
                }
            }
        }
    }
}

#[test]
fn linearly_weighted_segment_of_square_pyramidal_numbers() {
    // sum_{k=1}^{n} (n - k + 1) Q(m + k - 1, 1)
    //     == Q(n, 3) + 2(m - 1) S(n, 3) + (m - 1)^2 S(n, 2) + Q(m - 1, 1) S(n, 1)
    // and adding n * Q(m - 1, 2) gives the plain segment of Q(., 2).
    for m in 1..=10u64 {
        for n in 0..=10u64 {
            let p = m - 1;
            let weighted: u128 = (1..=n)
                .map(|k| (n - k + 1) as u128 * naive_iterated_square_sum(m + k - 1, 1))
                .sum();
            let expansion = naive_iterated_square_sum(n, 3)
                + 2 * p as u128 * simplex_number(n, 3)
                + (p * p) as u128 * simplex_number(n, 2)
                + naive_iterated_square_sum(p, 1) * simplex_number(n, 1);
            assert_eq!(weighted, expansion, "m={m} n={n}");
            assert_eq!(
                iterated_segment_iterated_square_sum(m, n, 1, 2),
                expansion,
                "m={m} n={n}"
            );
            assert_eq!(
                expansion + n as u128 * naive_iterated_square_sum(p, 2),
                segment_iterated_square_sum(m, n, 2),
                "m={m} n={n}"
            );
        }
    }
}

#[test]
fn segment_of_iterated_square_sums_needs_every_offset_order() {
    // sum_{k=m}^{m+n-1} Q(k, r) == Q(n, r+1) + 2(m-1) S(n, r+1)
    //     + sum_{j=0}^{r} Q(m-1, j) S(n, r-j)
    // Truncating the last sum after j = 2 is only correct for r <= 2.
    for r in 0..=5u64 {
        for m in 1..=8u64 {
            for n in 1..=8u64 {
                let p = m - 1;
                let head =
                    naive_iterated_square_sum(n, r + 1) + 2 * p as u128 * simplex_number(n, r + 1);
                let offset = |j: u64| {
                    if p == 0 && j == 0 {
                        0
                    } else if j == 0 {
                        (p * p) as u128
                    } else {
                        naive_iterated_square_sum(p, j)
                    }
                };
                let full: u128 = head
                    + (0..=r)
                        .map(|j| offset(j) * simplex_number(n, r - j))
                        .sum::<u128>();
                assert_eq!(
                    segment_iterated_square_sum(m, n, r),
                    full,
                    "r={r} m={m} n={n}"
                );

                let truncated: u128 = head
                    + (0..=r.min(2))
                        .map(|j| offset(j) * simplex_number(n, r - j))
                        .sum::<u128>();
                if r <= 2 || p == 0 {
                    assert_eq!(truncated, full, "r={r} m={m} n={n}");
                } else {
                    assert!(truncated < full, "r={r} m={m} n={n}");
                }
            }
        }
    }
}

#[test]
fn index_weighted_triangular_sum_identities() {
    // L(n) = sum_k k T(k), T(k) = k(k+1)/2
    for n in 0..=30u64 {
        let l: u128 = (1..=n).map(|k| k as u128 * pascal(k + 1, 2)).sum();
        assert_eq!(generalized_series_sum(n, 2), l, "L({n})");
        // n(n+1)(n+2)(3n+1)/24 == S(n, 2) (3n + 1) / 4
        assert_eq!(simplex_number(n, 2) * (3 * n + 1) as u128, 4 * l, "L({n})");
        // trapezoidal-number form: sum_k k (n + k)(n - k + 1) / 2
        let trapezoid: u128 = (1..=n)
            .map(|k| k as u128 * ((n + k) * (n - k + 1) / 2) as u128)
            .sum();
        assert_eq!(trapezoid, l, "L({n})");
        assert_eq!(trapezoid_weighted_segment_simplex_sum(1, n, 0), l, "L({n})");
        if n >= 1 {
            // S(n, 3) + 2 S(n - 1, 3)
            assert_eq!(
                simplex_number(n, 3) + 2 * simplex_number(n - 1, 3),
                l,
                "L({n})"
            );
        }
        if n >= 2 {
            // Newton form: S(n, 1) + 4 S(n - 1, 2) + 3 S(n - 2, 3)
            assert_eq!(
                simplex_number(n, 1) + 4 * simplex_number(n - 1, 2) + 3 * simplex_number(n - 2, 3),
                l,
                "L({n})"
            );
        }
    }
}

#[test]
fn generalized_series_sum_shift_identities() {
    for r in 0..=6u64 {
        for n in 1..=30u64 {
            let g = generalized_series_sum(n, r);
            assert_eq!(
                g,
                (r + 1) as u128 * simplex_number(n - 1, r + 1) + simplex_number(n, r),
                "G({n}, {r})"
            );
            assert_eq!(
                g,
                r as u128 * simplex_number(n - 1, r + 1) + simplex_number(n, r + 1),
                "G({n}, {r})"
            );
        }
    }
}

#[test]
fn segment_generalized_series_sum_matches_naive_loop() {
    for r in 0..=5u64 {
        for start in 0..=10u64 {
            for count in 0..=12u64 {
                let expected: u128 = (start..start + count)
                    .map(|k| {
                        let m = if k == 0 {
                            (r == 0) as u128
                        } else {
                            pascal(k + r - 1, r)
                        };
                        k as u128 * m
                    })
                    .sum();
                assert_eq!(
                    segment_generalized_series_sum(start, count, r),
                    expected,
                    "segment_generalized_series_sum({start}, {count}, {r})"
                );
            }
        }
    }
}

#[test]
fn segment_index_weighted_triangular_sum_expansions() {
    for m in 1..=10u64 {
        for n in 0..=10u64 {
            let p = m - 1;
            let (pu, nu) = (p as u128, n as u128);
            let l = generalized_series_sum(n, 2);
            let t = |x: u64| simplex_number(x, 1);

            // sum_{k=m}^{m+n-1} (k - m + 1) T(k)
            //     == L(n) + 2(m-1) S(n, 2) + (T(m) + 1 - 2m) S(n, 1)
            let ascending = ascending_weighted_segment_simplex_sum(m, n, 1);
            assert_eq!(
                ascending as i128,
                (l + 2 * pu * simplex_number(n, 2)) as i128
                    + (t(m) as i128 + 1 - 2 * m as i128) * simplex_number(n, 1) as i128,
                "m={m} n={n}"
            );

            // sum_{k=m}^{m+n-1} k T(k)
            //     == L(n) + 3(m-1) S(n, 2) + [(m-1)^2 + T(m-1) - (m-1)] S(n, 1)
            //      + (m-1) T(m-1) n
            let absolute = segment_generalized_series_sum(m, n, 2);
            assert_eq!(
                absolute,
                l + 3 * pu * simplex_number(n, 2)
                    + (pu * pu + t(p) - pu) * simplex_number(n, 1)
                    + pu * t(p) * nu,
                "m={m} n={n}"
            );
            assert_eq!(
                absolute,
                pu * segment_simplex_sum(m, n, 1) + ascending,
                "m={m} n={n}"
            );

            if n >= 2 {
                // Newton forms with leading differences of the trapezoid-weighted
                // sum: m, 2(m + 1), 3.
                let trapezoid = trapezoid_weighted_segment_simplex_sum(m, n, 0);
                let newton = m as u128 * simplex_number(n, 1)
                    + 2 * (m + 1) as u128 * simplex_number(n - 1, 2)
                    + 3 * simplex_number(n - 2, 3);
                assert_eq!(trapezoid, newton, "m={m} n={n}");
                assert_eq!(
                    newton_iterated_sum(m as i128, &[2 * (m as i128 + 1), 3], n, 2),
                    newton as i128,
                    "m={m} n={n}"
                );
                // (m - 1) times the sum of squares separates it from L(n).
                assert_eq!(trapezoid, l + pu * iterated_square_sum(n, 1), "m={m} n={n}");

                // sum_k k T(m + k - 1) == T(m) S(n, 1) + 2(m+1) S(n-1, 2) + 3 S(n-2, 3)
                assert_eq!(
                    ascending,
                    t(m) * simplex_number(n, 1)
                        + 2 * (m + 1) as u128 * simplex_number(n - 1, 2)
                        + 3 * simplex_number(n - 2, 3),
                    "m={m} n={n}"
                );
                assert_eq!(
                    ascending,
                    t(p) * simplex_number(n, 1) + trapezoid,
                    "m={m} n={n}"
                );
            }
        }
    }
}

#[test]
fn position_weighted_segment_sums_match_naive_loop() {
    for d in 0..=4u64 {
        for start in 0..=8u64 {
            for count in 0..=10u64 {
                let term = |i: u64| pascal(start + i + d, d + 1);
                let ascending: u128 = (0..count).map(|i| (i + 1) as u128 * term(i)).sum();
                let trapezoid: u128 = (0..count)
                    .map(|i| term(i) * (i + 1..=count).map(|k| k as u128).sum::<u128>())
                    .sum();
                assert_eq!(
                    ascending_weighted_segment_simplex_sum(start, count, d),
                    ascending,
                    "ascending_weighted_segment_simplex_sum({start}, {count}, {d})"
                );
                assert_eq!(
                    trapezoid_weighted_segment_simplex_sum(start, count, d),
                    trapezoid,
                    "trapezoid_weighted_segment_simplex_sum({start}, {count}, {d})"
                );
            }
        }
    }
}

fn pow(base: u64, exp: u64) -> u128 {
    (0..exp).fold(1u128, |acc, _| acc * base as u128)
}

#[test]
fn stirling_second_kind_matches_surjection_count() {
    // k! S(n, k) == sum_i (-1)^i C(k, i) (k - i)^n (inclusion-exclusion).
    for n in 0..=12u64 {
        for k in 0..=14u64 {
            let mut signed: i128 = 0;
            for i in 0..=k {
                let term = (pascal(k, i) * pow(k - i, n)) as i128;
                signed += if i % 2 == 0 { term } else { -term };
            }
            let factorial: u128 = (1..=k as u128).product();
            assert_eq!(
                stirling_second_kind(n, k) * factorial,
                signed as u128,
                "S({n}, {k})"
            );
        }
    }
}

#[test]
fn power_forward_differences_match_difference_table() {
    for power in 0..=7u64 {
        for start in 0..=10u64 {
            let values: Vec<i128> = (0..=power + 2)
                .map(|i| pow(start + i, power) as i128)
                .collect();
            let table = leading_differences(&values);
            let expected: Vec<u128> = table[..=power as usize]
                .iter()
                .map(|&d| d as u128)
                .collect();
            assert_eq!(
                power_forward_differences(start, power),
                expected,
                "power_forward_differences({start}, {power})"
            );
            assert!(table[power as usize + 1..].iter().all(|&d| d == 0));
        }
    }
    // Cubes: m^3, 3m^2 + 3m + 1, 6m + 6, 6
    for m in 0..=20u64 {
        let mu = m as u128;
        assert_eq!(
            power_forward_differences(m, 3),
            vec![mu * mu * mu, 3 * mu * mu + 3 * mu + 1, 6 * mu + 6, 6]
        );
    }
}

#[test]
fn iterated_segment_power_sum_matches_naive_loop() {
    for power in 0..=6u64 {
        for depth in 0..=4u64 {
            for start in 0..=8u64 {
                for count in 0..=10u64 {
                    let terms = (start..start + count).map(|k| pow(k, power)).collect();
                    assert_eq!(
                        iterated_segment_power_sum(start, count, power, depth),
                        iterate_partial_sums(terms, depth),
                        "iterated_segment_power_sum({start}, {count}, {power}, {depth})"
                    );
                }
            }
        }
    }
}

#[test]
fn iterated_power_sum_generalizes_squares() {
    for n in 0..=25u64 {
        for depth in 0..=5u64 {
            assert_eq!(
                iterated_power_sum(n, 2, depth),
                iterated_square_sum(n, depth),
                "n={n} depth={depth}"
            );
        }
        assert_eq!(power_sum(n, 1), simplex_number(n, 1));
    }
}

#[test]
fn cube_sum_identities() {
    for n in 0..=40u64 {
        let t = simplex_number(n, 1);
        // sum k^3 == T(n)^2
        assert_eq!(power_sum(n, 3), t * t, "n={n}");
        // sum (n - k + 1) k^3 == S(n, 2) (3n^2 + 6n + 1) / 10
        assert_eq!(
            iterated_power_sum(n, 3, 2) * 10,
            simplex_number(n, 2) * (3 * n * n + 6 * n + 1) as u128,
            "n={n}"
        );
        let weighted: u128 = (1..=n).map(|k| (n - k + 1) as u128 * pow(k, 3)).sum();
        assert_eq!(iterated_power_sum(n, 3, 2), weighted, "n={n}");
        if n >= 3 {
            let s = |x: u64, d: u64| simplex_number(x, d);
            // sum k^3 == n + 7 S(n-1, 1) + 12 S(n-2, 2) + 6 S(n-3, 3)
            assert_eq!(
                power_sum(n, 3),
                n as u128 + 7 * s(n - 1, 1) + 12 * s(n - 2, 2) + 6 * s(n - 3, 3),
                "n={n}"
            );
            // sum (n - k + 1) k^3 == S(n, 1) + 7 S(n-1, 2) + 12 S(n-2, 3) + 6 S(n-3, 4)
            assert_eq!(
                iterated_power_sum(n, 3, 2),
                s(n, 1) + 7 * s(n - 1, 2) + 12 * s(n - 2, 3) + 6 * s(n - 3, 4),
                "n={n}"
            );
        }
    }
}

#[test]
fn segment_cube_sum_difference_expansions() {
    let s = |x: u64, d: u64| simplex_number(x, d);
    for m in 0..=12u64 {
        for n in 3..=12u64 {
            let mu = m as u128;
            let (d0, d1, d2, d3) = (mu * mu * mu, 3 * mu * mu + 3 * mu + 1, 6 * mu + 6, 6);
            // sum_{k=m}^{m+n-1} k^3
            //     == m^3 n + (3m^2+3m+1) S(n-1, 1) + (6m+6) S(n-2, 2) + 6 S(n-3, 3)
            assert_eq!(
                iterated_segment_power_sum(m, n, 3, 1),
                d0 * n as u128 + d1 * s(n - 1, 1) + d2 * s(n - 2, 2) + d3 * s(n - 3, 3),
                "m={m} n={n}"
            );
            // sum_{k=1}^{n} (n-k+1)(m+k-1)^3
            //     == m^3 S(n, 1) + (3m^2+3m+1) S(n-1, 2) + (6m+6) S(n-2, 3) + 6 S(n-3, 4)
            assert_eq!(
                iterated_segment_power_sum(m, n, 3, 2),
                d0 * s(n, 1) + d1 * s(n - 1, 2) + d2 * s(n - 2, 3) + d3 * s(n - 3, 4),
                "m={m} n={n}"
            );
        }
    }
}

#[test]
fn simplex_convolution_matches_naive_loop() {
    for a in 0..=4u64 {
        for b in 0..=4u64 {
            for n in 0..=15u64 {
                let expected: u128 = (1..=n)
                    .map(|k| pascal(n - k + 1 + a, a + 1) * pascal(k + b, b + 1))
                    .sum();
                assert_eq!(simplex_convolution(n, a, b), expected, "n={n} a={a} b={b}");
            }
        }
    }
}

#[test]
fn triangular_products_lemma() {
    // sum_k T(n-k+1) T(k) == sum_k k (T(1) + ... + T(n-k+1))
    //                     == sum_k (n-k+1) S(k, 2) == S(n, 4)
    for n in 0..=25u64 {
        let t = |x: u64| simplex_number(x, 1);
        let products: u128 = (1..=n).map(|k| t(n - k + 1) * t(k)).sum();
        let index_weighted: u128 = (1..=n)
            .map(|k| k as u128 * (1..=n - k + 1).map(t).sum::<u128>())
            .sum();
        let linear: u128 = (1..=n)
            .map(|k| (n - k + 1) as u128 * simplex_number(k, 2))
            .sum();
        assert_eq!(products, simplex_number(n, 4), "n={n}");
        assert_eq!(index_weighted, products, "n={n}");
        assert_eq!(linear, products, "n={n}");
        // sum_k k S(k, 2) == S(n, 4) + 3 S(n - 1, 4)
        if n >= 1 {
            assert_eq!(
                generalized_series_sum(n, 3),
                simplex_number(n, 4) + 3 * simplex_number(n - 1, 4),
                "n={n}"
            );
        }
    }
}

#[test]
fn trapezoid_weighted_tetrahedral_newton_forms() {
    let s = |x: u64, d: u64| simplex_number(x, d);
    for m in 1..=12u64 {
        for n in 3..=12u64 {
            let p = m - 1;
            // sum_k (T(1) + ... + T(m + k - 1)) (k + ... + n) == sum_k k * sum_{j=m}^{m+k-1} T(j)
            let trapezoid = trapezoid_weighted_segment_simplex_sum(m, n, 1);
            let running: u128 = (1..=n)
                .map(|k| k as u128 * segment_simplex_sum(m, k, 1))
                .sum();
            assert_eq!(trapezoid, running, "m={m} n={n}");
            // == L2(n) + (m-1) L1(n) + T(m-1) Q(n, 1)
            assert_eq!(
                trapezoid,
                generalized_series_sum(n, 3)
                    + p as u128 * generalized_series_sum(n, 2)
                    + s(p, 1) * iterated_square_sum(n, 1),
                "m={m} n={n}"
            );
            // Newton form with differences T(m), 2 T(m+1), 3(m+2), 4 at depth 2.
            let newton = s(m, 1) * s(n, 1)
                + 2 * s(m + 1, 1) * s(n - 1, 2)
                + 3 * (m + 2) as u128 * s(n - 2, 3)
                + 4 * s(n - 3, 4);
            assert_eq!(trapezoid, newton, "m={m} n={n}");
            // sum_k k S(m + k - 1, 2): replace T(m) by S(m, 2) in the first term.
            assert_eq!(
                ascending_weighted_segment_simplex_sum(m, n, 2),
                newton - s(m, 1) * s(n, 1) + s(m, 2) * s(n, 1),
                "m={m} n={n}"
            );
            assert_eq!(
                ascending_weighted_segment_simplex_sum(m, n, 2),
                trapezoid + s(p, 2) * s(n, 1),
                "m={m} n={n}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Application problems of the source text ([30]-[41]): the stated totals are
// recomputed with library functions (and naive loops), and the published
// polynomial equations are solved exactly with positive_integer_root.
// ---------------------------------------------------------------------------

use coolsigma::roots::positive_integer_root;

/// Naive cumulative-crew wage model: on day j the wage is `wage(j)` and the
/// crew is the running total of `recruit(1..=j)`.
fn naive_cumulative_wage_sum(
    days: u64,
    recruit: impl Fn(u64) -> u128,
    wage: impl Fn(u64) -> u128,
) -> u128 {
    let mut crew = 0u128;
    let mut total = 0u128;
    for j in 1..=days {
        crew += recruit(j);
        total += wage(j) * crew;
    }
    total
}

/// Naive remaining-days convolution: each recruit batch earns a fixed wage
/// for every remaining day.
fn naive_remaining_days_sum(days: u64, recruit: impl Fn(u64) -> u128, wage: u128) -> u128 {
    (1..=days)
        .map(|k| wage * recruit(k) * (days - k + 1) as u128)
        .sum()
}

#[test]
fn problem_30_recruiting_workers() {
    // p_k = 45 + 5k, 3 doe per day, R = 15980 in the source summary; the
    // summary's own equation 2(R/3) = 31980 pins the consistent value 15990.
    let r_over_3 = 45 * simplex_number(12, 1) + 5 * simplex_number(12, 2);
    assert_eq!(3 * r_over_3, 15990);
    // Naive convolution check.
    assert_eq!(
        naive_remaining_days_sum(12, |k| (45 + 5 * k) as u128, 3),
        15990
    );
    assert_eq!(positive_integer_root(&[-6396, 29, 30, 1]), Some(12));
}

#[test]
fn problem_31_shrinking_crew_time_reversed() {
    // p_k = 90 - 4(k - 1), 3 doe per day, R = 25680.
    let r_over_3 = (90u128 - 4 * 15) * simplex_number(15, 1) + 4 * iterated_square_sum(15, 1);
    assert_eq!(3 * r_over_3, 25680);
    assert_eq!(
        naive_remaining_days_sum(15, |k| (94 - 4 * k) as u128, 3),
        25680
    );
    assert_eq!(positive_integer_root(&[25680, -137, -135, 2]), Some(15));
}

#[test]
fn problem_32_rising_price_triangular_pile() {
    assert_eq!(simplex_number(7, 2) + 2 * simplex_number(7, 3), 504);
    let naive: u128 = (1..=7)
        .map(|k| (2 * k + 1) as u128 * pascal(7 - k + 2, 2))
        .sum();
    assert_eq!(naive, 504);
    assert_eq!(positive_integer_root(&[-6048, 10, 17, 8, 1]), Some(7));
}

#[test]
fn problem_33_rising_price_square_pile() {
    assert_eq!(
        2 * iterated_square_sum(11, 2) - iterated_square_sum(11, 1),
        2926
    );
    let naive: u128 = (1..=11)
        .map(|k| (2 * k - 1) as u128 * ((11 - k + 1) as u128).pow(2))
        .sum();
    assert_eq!(naive, 2926);
    assert_eq!(positive_integer_root(&[-17556, 1, 2, 2, 1]), Some(11));
}

#[test]
fn problem_34_cube_weighted_sum() {
    // sum (2k) k^2 = 2 * (sum k^3) = 4050, n(n + 1) = 90.
    assert_eq!(2 * power_sum(9, 3), 4050);
    assert_eq!(9 * 10, 90);
}

#[test]
fn problem_35_truncated_triangular_pile() {
    // sum (2k + 3) S_{k+2}^1 = 2 * asc(3, n, 1) + 3 * segment(3, n, 1).
    let value =
        2 * ascending_weighted_segment_simplex_sum(3, 7, 1) + 3 * segment_simplex_sum(3, 7, 1);
    assert_eq!(value, 2135);
    let naive: u128 = (1..=7)
        .map(|k| (2 * k + 3) as u128 * simplex_number(k + 2, 1))
        .sum();
    assert_eq!(naive, 2135);
    assert_eq!(positive_integer_root(&[-25620, 202, 123, 32, 3]), Some(7));
}

#[test]
fn problem_36_double_growth_convolution() {
    // Crew 30 + 10(k - 1), wage 10 + (k - 1) per day, total 8350 pun.
    // NB: the source summary writes sum p_k w_k (n - k + 1), which evaluates
    // to 7600; the model consistent with its own quartic is the cumulative
    // crew times the day-of-work wage.
    assert_eq!(
        naive_cumulative_wage_sum(5, |k| (20 + 10 * k) as u128, |j| (j + 9) as u128),
        8350
    );
    assert_eq!(positive_integer_root(&[-20040, 298, 357, 62, 3]), Some(5));
}

#[test]
fn problem_37_reverse_weighted_grain_piles() {
    // sum (n - k + 1) T^1_{k+2} with T^1_x = 2 S_x^2 - S_x^1.
    let value = 2 * weighted_segment_simplex_sum(3, 7, 2) - weighted_segment_simplex_sum(3, 7, 1);
    assert_eq!(value, 2030);
    let naive: u128 = (1..=7)
        .map(|k| (8 - k) as u128 * iterated_square_sum(k + 2, 1))
        .sum();
    assert_eq!(naive, 2030);
    assert_eq!(
        positive_integer_root(&[-243600, 598, 805, 240, 35, 2]),
        Some(7)
    );
}

#[test]
fn problem_38_area_proportional_recruiting() {
    // R/3 = sum_j j * sum_{k=1}^j (k + 4)^2 == weighted_shifted_square_sum(5, N)
    // with N = 15 days (the summary's polynomial is already in N).
    assert_eq!(3 * weighted_shifted_square_sum(5, 15), 447732);
    let naive = naive_cumulative_wage_sum(15, |k| ((k + 4) as u128).pow(2), |j| 3 * j as u128);
    assert_eq!(naive, 447732);
    assert_eq!(
        positive_integer_root(&[-17909280, 402, 1345, 1090, 155, 8]),
        Some(15)
    );
}

#[test]
fn problem_39_cubic_volume_recruiting() {
    // R/250 = sum_j sum_{k=1}^j (k + 2)^3, cumulative crews of cubic size,
    // N = 15 days. Simplex-basis expansion from the source:
    let expansion = 27 * simplex_number(15, 1)
        + 37 * simplex_number(14, 2)
        + 24 * simplex_number(13, 3)
        + 6 * simplex_number(12, 4);
    assert_eq!(250 * expansion, 23_462_000);
    // Same value via Nicomedes: sum_{j<=N} (T_{j+2})^2 - 9N
    //   == iterated_power_sum(N + 2, 3, 2) - 10 - 9N.
    assert_eq!(
        250 * (iterated_power_sum(17, 3, 2) - 10 - 9 * 15),
        23_462_000
    );
    let naive = naive_cumulative_wage_sum(15, |k| ((k + 2) as u128).pow(3), |_| 250);
    assert_eq!(naive, 23_462_000);
    // Reference polynomial (variable already shifted to N = 15).
    assert_eq!(
        positive_integer_root(&[-11261760, 1084, 1530, 530, 90, 6]),
        Some(15)
    );
    // Comparison-text polynomials in the shifted variable n = N - 3.
    assert_eq!(positive_integer_root(&[-92736, 660, 181, 22, 1]), Some(12));
    assert_eq!(
        positive_integer_root(&[-5610840, 18362, 6390, 1075, 90, 3]),
        Some(12)
    );
}

#[test]
fn problem_40_cone_pile_pair() {
    // Even pile (14 layers) + odd pile (13 layers), reverse price weights.
    let total = iterated_interleaved_series_sum(14) + iterated_interleaved_series_sum(13);
    assert_eq!(total, 5565);
    // Naive: W_i running totals of the interleaved layer counts.
    let mut layer = 0u128;
    let mut total14 = 0u128;
    for i in 1..=14u64 {
        layer += if i % 2 == 1 {
            3 * i.div_ceil(2).pow(2) as u128 - 3 * i.div_ceil(2) as u128 + 1
        } else {
            3 * (i / 2).pow(2) as u128
        };
        total14 += layer;
    }
    assert_eq!(total14, iterated_interleaved_series_sum(14));
    assert_eq!(positive_integer_root(&[-44520, 2, 3, 2, 1]), Some(14));
}

#[test]
fn problem_41_two_unknowns_resultant() {
    // Earthworkers n = 16 days, carpenters m = 12 days.
    assert_eq!(simplex_number(16, 1), 12 * 12 - 8);
    assert_eq!(simplex_number(16, 2) + iterated_square_sum(12, 1), 1466);
    // NB: the source summary's sextic coefficients contain transcription
    // errors (its n-polynomial swaps the n^5/n^4 coefficients and flips the
    // n^3 sign; its m-polynomial flips the m^5 sign). The polynomials below
    // are the true Sylvester resultants of the pair
    //   2m^2 - n^2 - n - 16 == 0,
    //   2m^3 + 3m^2 + m + n^3 + 3n^2 + 2n - 8796 == 0,
    // recomputed independently; both vanish at the documented solution.
    let n_poly: [i128; 7] = [615565376, -494564, -635018, -140504, 6, 60, 4];
    let m_poly: [i128; 7] = [-77796432, 17640, 107359, 35268, -231, -24, 4];
    assert_eq!(positive_integer_root(&n_poly), Some(16));
    assert_eq!(positive_integer_root(&m_poly), Some(12));
}

#[test]
fn iterated_interleaved_series_sum_matches_naive_loop() {
    for n in 1..=60u64 {
        let mut layer = 0u128;
        let mut expected = 0u128;
        for i in 1..=n {
            layer += if i % 2 == 1 {
                let m = i.div_ceil(2) as u128;
                3 * m * m - 3 * m + 1
            } else {
                let m = (i / 2) as u128;
                3 * m * m
            };
            expected += layer;
        }
        assert_eq!(
            iterated_interleaved_series_sum(n),
            expected,
            "iterated_interleaved_series_sum({n})"
        );
    }
}
