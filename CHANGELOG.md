# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [1.0.1]

### Added

- `#![forbid(unsafe_code)]` on the crate root.
- This changelog.
- README installation section now matches the released version.

## [1.0.0]

First public release.

### Features

- `combinations`: binomial coefficients, multiset coefficients, Stirling
  numbers of the second kind.
- `sums`: simplex numbers and their segment, weighted (ascending, descending,
  trapezoidal), Vandermonde, iterated, and convolution sums.
- `series`: generalized combinatorial series sums, arithmetic series sums,
  centered polygonal expansions, iterated power/square sums and their
  segment/offset-expansion variants, rectangular frustum sums, weighted
  partial/shifted square sums, and interleaved cone-pile sums with their
  double accumulation.
- `interpolation`: forward differences, Newton forward-difference
  interpolation (`f64`), exact integer difference tables and iterated sums
  (`i128`).
- `roots`: exact positive integer roots of integer polynomials via a
  divisor-enumeration / Newton-Raphson hybrid.

### Guarantees

- All `combinations` / `sums` / `series` functions take `u64` inputs and
  return exact `u128` results, with gcd-based factor cancellation to avoid
  intermediate overflow.
- Every such function `f` has a `checked_f` variant returning
  `Option<u128>` for pipelines that must not panic.
- Every closed form is cross-checked against independent brute-force
  reference implementations (Pascal's triangle, direct summation), and the
  twelve application problems of the source text are encoded end to end in
  `tests/reference.rs`.
