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
use coolsigma::combinations::combinations;
use coolsigma::series::{arithmetic_series_sum, iterated_square_sum};
use coolsigma::sums::simplex_sum;

// Binomial coefficient C(10, 3) = 120
assert_eq!(combinations(10, 3), 120);

// Tetrahedral number: 1 + 3 + 6 + 10 = 20
assert_eq!(simplex_sum(2, 4), 20);

// Double sum of squares: sum_{j=1}^{4} sum_{k=1}^{j} k^2 = 1 + 5 + 14 + 30 = 50
assert_eq!(iterated_square_sum(4, 2), 50);

// Segment sum 5 + 6 + 7 + 8 = 26, generalized with a common difference
assert_eq!(arithmetic_series_sum(5, 4, 1), 26);
assert_eq!(arithmetic_series_sum(1, 4, 2), 16); // 1 + 3 + 5 + 7
```

## API overview

| Module | Function | Computes |
|---|---|---|
| `combinations` | `combinations(n, k)` | Binomial coefficient `C(n, k)` |
| `sums` | `simplex_sum(n, k)` | Simplex sum `C(n + k, n + 1)` |
| `sums` | `truncated_simplex_sum(start, n, k)` | Simplex sum over `[start, k]` |
| `sums` | `simplex_number(n, dimension)` | `C(n + dimension, dimension + 1)` |
| `sums` | `segment_simplex_sum(start, count, dimension)` | Segment sum of consecutive simplex numbers |
| `series` | `generalized_series_sum(n, r)` | `prod (n+i) * ((r+1)n + 1) / (r+2)!` |
| `series` | `arithmetic_series_sum(first, count, diff)` | Sum of an arithmetic sequence |
| `series` | `centered_expansion(sides, layer)` | Centered polygonal number `1 + sides * T(layer)` |
| `series` | `iterated_square_sum(n, depth)` | `depth`-fold iterated sum of squares |
| `series` | `weighted_partial_square_sum(n)` | `sum k * (1^2 + ... + k^2)` |
| `series` | `interleaved_series_sum(n)` | Interleaved hexagonal/triple-square sum |
| `interpolation` | `forward_difference(prev, curr)` | `curr - prev` |
| `interpolation` | `newton_forward_interpolation(base, s, deltas)` | Newton forward-difference interpolation |

The `interpolation` module is the deliberate exception to the `u64 -> u128`
contract: `forward_difference` takes `u128` terms (it is applied to the
outputs of the functions above), and `newton_forward_interpolation` operates
on `f64` because interpolation is real-valued by definition.

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
