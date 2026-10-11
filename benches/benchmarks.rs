//! Criterion benchmarks for the closed-form evaluation paths.
//!
//! Run with `cargo bench`. Parameters are chosen so every intermediate
//! result stays inside `u128` (results larger than ~3.4e38 panic by design).

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use coolsigma::combinations::{combinations, stirling_second_kind};
use coolsigma::interpolation::newton_iterated_sum;
use coolsigma::roots::positive_integer_root;
use coolsigma::series::{iterated_power_sum, power_sum, weighted_shifted_pyramidal_sum};
use coolsigma::sums::{segment_simplex_sum, simplex_number};

/// Cost of a closed form is constant in the summation bound n: only the
/// fixed number of multiplications for the degree matters.
fn bench_simplex_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex_number_dim3");
    for n in [10u64, 1_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| simplex_number(black_box(n), black_box(3)));
        });
    }
    group.finish();
}

/// Cost of a power sum grows with the exponent (Stirling expansion length),
/// not with n.
fn bench_power_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_sum_n1000");
    for p in [2u64, 6, 11] {
        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, &p| {
            b.iter(|| power_sum(black_box(1_000), black_box(p)));
        });
    }
    group.finish();
}

/// Binomial coefficients via gcd-cancelled factor products.
fn bench_combinations(c: &mut Criterion) {
    let mut group = c.benchmark_group("combinations");
    for (n, k) in [(100u64, 50u64), (126, 63)] {
        group.bench_with_input(BenchmarkId::new("central", n), &(n, k), |b, &(n, k)| {
            b.iter(|| combinations(black_box(n), black_box(k)));
        });
    }
    group.finish();
}

fn bench_stirling(c: &mut Criterion) {
    c.bench_function("stirling_second_kind_20_10", |b| {
        b.iter(|| stirling_second_kind(black_box(20), black_box(10)));
    });
}

/// Representative multi-depth and weighted closed forms.
fn bench_compound_sums(c: &mut Criterion) {
    let mut group = c.benchmark_group("compound_sums");
    group.bench_function("iterated_power_sum(n=10^4, p=2, d=3)", |b| {
        b.iter(|| iterated_power_sum(black_box(10_000), black_box(2), black_box(3)));
    });
    group.bench_function("weighted_shifted_pyramidal_sum(m=10, n=10^4)", |b| {
        b.iter(|| weighted_shifted_pyramidal_sum(black_box(10), black_box(10_000)));
    });
    group.bench_function("segment_simplex_sum(start=100, count=10^6, dim=4)", |b| {
        b.iter(|| segment_simplex_sum(black_box(100), black_box(1_000_000), black_box(4)));
    });
    group.finish();
}

/// Both root-finding stages: divisor enumeration (small constant term) and
/// the Newton–Raphson path (constant term above the 10^12 threshold).
fn bench_roots(c: &mut Criterion) {
    let mut group = c.benchmark_group("positive_integer_root");
    // x^3 + 30x^2 + 29x - 6396, root 12: divisor enumeration stage.
    group.bench_function("enumeration", |b| {
        b.iter(|| positive_integer_root(black_box(&[-6396, 29, 30, 1])));
    });
    // (x - 1000003)(x + 20000000), root 1000003: Newton stage.
    group.bench_function("newton", |b| {
        b.iter(|| positive_integer_root(black_box(&[-20_000_060_000_000, 18_999_997, 1])));
    });
    group.finish();
}

/// Integer Newton forward evaluation (the exact alternative to the f64
/// interpolation path).
fn bench_iterated_sum(c: &mut Criterion) {
    c.bench_function("newton_iterated_sum(count=10^3, depth=2)", |b| {
        b.iter(|| {
            newton_iterated_sum(
                black_box(0),
                black_box(&[1, 6, 6]),
                black_box(1_000),
                black_box(2),
            )
        });
    });
}

/// Closed form against a naive accumulation loop for sum k^2 over 10^6
/// terms: the measured O(1) versus O(n) contrast.
fn bench_closed_form_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_of_squares_n10^6");
    group.bench_function("closed_form", |b| {
        b.iter(|| power_sum(black_box(1_000_000), black_box(2)));
    });
    group.bench_function("naive_loop", |b| {
        b.iter(|| {
            let mut acc = 0u128;
            for k in 1..=1_000_000u128 {
                acc += k * k;
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_simplex_scaling,
    bench_power_sum,
    bench_combinations,
    bench_stirling,
    bench_compound_sums,
    bench_roots,
    bench_iterated_sum,
    bench_closed_form_vs_naive,
);
criterion_main!(benches);
