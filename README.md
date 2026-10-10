# coolsigma

Closed-form combinatorial and polynomial series sums with exact `u128` arithmetic.

Every function in `combinations`, `sums`, and `series` takes `u64` inputs and
returns an exact `u128` result. Intermediate values are kept small by
cancelling common factors before multiplying, so overflow is avoided wherever
the result itself fits. Functions panic when the mathematical result does not
fit in `u128`.

## Installation

```toml
[dependencies]
coolsigma = "0.1"
```

## Quick start

```rust
use coolsigma::combinations::{combinations, multiset_coefficient};
use coolsigma::series::{
    arithmetic_series_sum, iterated_segment_square_sum, iterated_square_sum,
    rectangular_frustum_sum,
};
use coolsigma::sums::{
    segment_simplex_sum_vandermonde, simplex_sum, weighted_segment_simplex_sum,
};

// Binomial coefficient C(10, 3) = 120
assert_eq!(combinations(10, 3), 120);

// Multiset coefficient C(4 + 3 - 1, 3): size-3 multisets from 4 kinds
assert_eq!(multiset_coefficient(4, 3), 20);

// Tetrahedral number: 1 + 3 + 6 + 10 = 20
assert_eq!(simplex_sum(2, 4), 20);

// Double sum of squares: sum_{j=1}^{4} sum_{k=1}^{j} k^2 = 1 + 5 + 14 + 30 = 50
assert_eq!(iterated_square_sum(4, 2), 50);

// Segment sum 5 + 6 + 7 + 8 = 26, generalized with a common difference
assert_eq!(arithmetic_series_sum(5, 4, 1), 26);
assert_eq!(arithmetic_series_sum(1, 4, 2), 16); // 1 + 3 + 5 + 7

// Triangular numbers T(4) + ... + T(8) via the Chu-Vandermonde convolution:
// M(5,3) + M(3,1) M(5,2) + M(3,2) M(5,1) = 35 + 45 + 30
assert_eq!(segment_simplex_sum_vandermonde(4, 5, 1), 110);

// Linearly weighted segment: 3*T(2) + 2*T(3) + 1*T(4) = 9 + 12 + 10
assert_eq!(weighted_segment_simplex_sum(2, 3, 1), 31);

// Stepped frustum with a 2 x 4 top and 3 layers: 2*4 + 3*5 + 4*6
assert_eq!(rectangular_frustum_sum(2, 4, 3), 47);

// Sum of running totals of 3^2, 4^2, 5^2: 3*9 + 2*16 + 1*25
assert_eq!(iterated_segment_square_sum(3, 3, 2), 84);
```

## API overview

| Module | Function | Computes |
|---|---|---|
| `combinations` | `combinations(n, k)` | Binomial coefficient `C(n, k)` |
| `combinations` | `multiset_coefficient(n, k)` | Multiset coefficient `C(n + k - 1, k)` |
| `combinations` | `stirling_second_kind(n, k)` | Stirling number of the second kind `S(n, k)` |
| `sums` | `simplex_sum(n, k)` | Simplex sum `C(n + k, n + 1)` |
| `sums` | `truncated_simplex_sum(start, n, k)` | Simplex sum over `[start, k]` |
| `sums` | `simplex_number(n, dimension)` | `C(n + dimension, dimension + 1)` |
| `sums` | `segment_simplex_sum(start, count, dimension)` | Segment sum of consecutive simplex numbers |
| `sums` | `segment_simplex_sum_vandermonde(start, count, dimension)` | Same segment sum via the Chu–Vandermonde convolution |
| `sums` | `weighted_segment_simplex_sum(start, count, dimension)` | Segment sum with linearly decreasing weights `count, ..., 1` |
| `sums` | `simplex_convolution(n, a, b)` | `sum S(n-k+1, a) S(k, b) = S(n, a + b + 2)` |
| `sums` | `ascending_weighted_segment_simplex_sum(start, count, dimension)` | Segment sum with ascending weights `1, 2, ..., count` |
| `sums` | `trapezoid_weighted_segment_simplex_sum(start, count, dimension)` | Segment sum weighted by trapezoidal numbers `(i+1) + ... + count` |
| `sums` | `iterated_segment_simplex_sum(start, count, dimension, depth)` | `depth`-fold iterated partial sum of a simplex segment (multiset-coefficient weights) |
| `series` | `generalized_series_sum(n, r)` | `sum k * C(k+r-1, r) = prod (n+i) * ((r+1)n + 1) / (r+2)!` |
| `series` | `segment_generalized_series_sum(start, count, r)` | Segment of `k * C(k+r-1, r)` |
| `series` | `arithmetic_series_sum(first, count, diff)` | Sum of an arithmetic sequence |
| `series` | `centered_expansion(sides, layer)` | Centered polygonal number `1 + sides * T(layer)` |
| `series` | `iterated_square_sum(n, depth)` | `depth`-fold iterated sum of squares |
| `series` | `rectangular_frustum_sum(top_length, top_width, layers)` | `sum (a + i)(b + i)` via the discrete prismoidal formula |
| `series` | `segment_iterated_square_sum(start, count, order)` | Segment sum of consecutive iterated square sums |
| `series` | `segment_iterated_square_sum_offset_expansion(start, count, order)` | Same, expanded in the offset `start - 1` |
| `series` | `iterated_segment_iterated_square_sum(start, count, order, depth)` | `depth`-fold iterated partial sum of that segment |
| `series` | `iterated_segment_iterated_square_sum_offset_expansion(start, count, order, depth)` | Same, expanded in the offset `start - 1` |
| `series` | `iterated_segment_square_sum(start, count, depth)` | `depth`-fold iterated partial sum of a segment of squares (Newton expansion) |
| `series` | `power_sum(n, power)` | `1^p + 2^p + ... + n^p` |
| `series` | `iterated_power_sum(n, power, depth)` | `depth`-fold iterated power sum |
| `series` | `iterated_segment_power_sum(start, count, power, depth)` | `depth`-fold iterated partial sum of a segment of powers |
| `series` | `weighted_partial_square_sum(n)` | `sum k * (1^2 + ... + k^2)` |
| `series` | `interleaved_series_sum(n)` | Interleaved hexagonal/triple-square sum |
| `interpolation` | `forward_difference(prev, curr)` | `curr - prev` |
| `interpolation` | `newton_forward_interpolation(base, s, deltas)` | Newton forward-difference interpolation |
| `interpolation` | `power_forward_differences(start, power)` | Exact forward differences of `x^p` at `start` |
| `interpolation` | `leading_differences(values)` | Leading forward differences `[f(0), Δf(0), Δ²f(0), ...]` |
| `interpolation` | `newton_iterated_sum(base, deltas, count, depth)` | `depth`-fold iterated sum from a forward-difference table |

The `interpolation` module is the deliberate exception to the `u64 -> u128`
contract: `forward_difference` takes `u128` terms (it is applied to the
outputs of the functions above), `newton_forward_interpolation` operates
on `f64` because interpolation is real-valued by definition, and
`leading_differences` / `newton_iterated_sum` use `i128` because differences
of an arbitrary sequence can be negative.

## Correctness

All closed forms are cross-checked in `tests/reference.rs` against
independent brute-force implementations (Pascal's triangle, direct
summation) over grids of small inputs, alongside fixed-value unit tests and
documentation examples that run as doctests.

```sh
cargo test
```

## License

MIT
