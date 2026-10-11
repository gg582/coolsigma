//! Closed-form combinatorial and polynomial series sums.
//!
//! Every function in [`combinations`], [`sums`], and [`series`] takes `u64`
//! inputs and returns an exact `u128` result. Intermediate values are kept
//! small by cancelling common factors before multiplying, so overflow is
//! avoided wherever the result itself fits.
//!
//! Two deliberate exceptions to the `u64 -> u128` contract:
//!
//! - [`interpolation::forward_difference`] takes `u128` terms, since it is
//!   applied to the outputs of the functions above.
//! - [`interpolation::newton_forward_interpolation`] operates on `f64`,
//!   because interpolation is real-valued by definition.
//! - [`interpolation::leading_differences`] and
//!   [`interpolation::newton_iterated_sum`] operate on `i128`, because
//!   forward differences of an arbitrary sequence can be negative.
//! - [`roots::positive_integer_root`] takes `i128` coefficients, because
//!   polynomial coefficients can be negative.
//!
//! # Panics
//!
//! Functions panic when the mathematical result does not fit in `u128`.
//!
//! # Modules
//!
//! - [`combinations`]: binomial coefficients.
//! - [`sums`]: sums over simplex (binomial-coefficient) sequences.
//! - [`series`]: closed-form sums of polynomial series, including iterated
//!   sums of squares and stepped rectangular frustums.
//! - [`interpolation`]: finite differences, Newton forward-difference
//!   interpolation, and iterated sums from a difference table.
//! - [`roots`]: exact positive integer roots of polynomial equations.
//!
//! # Example
//!
//! ```
//! use coolsigma::series::iterated_square_sum;
//! use coolsigma::sums::simplex_number;
//!
//! assert_eq!(simplex_number(10, 2), 220); // 10th tetrahedral number
//! assert_eq!(iterated_square_sum(5, 2), 105); // double sum of squares
//! ```

mod common;

pub mod combinations;
pub mod interpolation;
pub mod roots;
pub mod series;
pub mod sums;

#[cfg(test)]
mod tests {
    use crate::combinations::{combinations, multiset_coefficient, stirling_second_kind};
    use crate::interpolation::{
        forward_difference, leading_differences, newton_forward_interpolation, newton_iterated_sum,
        power_forward_differences,
    };
    use crate::series::{
        arithmetic_series_sum, centered_expansion, centered_hexagonal_number,
        centered_triangular_number, generalized_series_sum, interleaved_series_sum,
        iterated_interleaved_series_sum, iterated_power_sum, iterated_segment_iterated_square_sum,
        iterated_segment_iterated_square_sum_offset_expansion, iterated_segment_power_sum,
        iterated_segment_square_sum, iterated_square_sum, odd_square, power_sum,
        rectangular_frustum_sum, segment_generalized_series_sum, segment_iterated_square_sum,
        segment_iterated_square_sum_offset_expansion, weighted_partial_square_sum,
        weighted_shifted_pyramidal_sum, weighted_shifted_square_sum,
    };
    use crate::sums::{
        ascending_weighted_segment_simplex_sum, iterated_segment_simplex_sum, segment_simplex_sum,
        segment_simplex_sum_vandermonde, simplex_convolution, simplex_number,
        trapezoid_weighted_segment_simplex_sum,
        weighted_segment_simplex_sum,
    };

    #[test]
    fn test_simplex_number() {
        assert_eq!(simplex_number(0, 1), 0);
        assert_eq!(simplex_number(10, 1), 55); // 1 + 2 + ... + 10
        assert_eq!(simplex_number(10, 2), 220); // tetrahedral
        assert_eq!(simplex_number(10, 3), 715);
        assert_eq!(simplex_number(5, 0), 5);
        assert_eq!(simplex_number(5, 1), 15);
        assert_eq!(simplex_number(5, 2), 35);
        assert_eq!(simplex_number(5, 3), 70); // pentatope
        assert_eq!(simplex_number(10, 100), 4_643_330_358_810);
    }

    #[test]
    fn test_segment_simplex_sum() {
        assert_eq!(segment_simplex_sum(4, 5, 0), 30); // 4 + 5 + 6 + 7 + 8
        assert_eq!(segment_simplex_sum(4, 3, 0), 15); // 4 + 5 + 6
        assert_eq!(segment_simplex_sum(1, 6, 0), 21); // 1 + ... + 6
        assert_eq!(segment_simplex_sum(4, 5, 1), 110);
        assert_eq!(segment_simplex_sum(4, 5, 2), 315);
        assert_eq!(segment_simplex_sum(1, 5, 2), 70); // 1 + 4 + 10 + 20 + 35
        assert_eq!(segment_simplex_sum(0, 4, 1), 10); // 0 + 1 + 3 + 6
        assert_eq!(segment_simplex_sum(7, 0, 3), 0); // empty segment
    }

    #[test]
    fn test_combinations() {
        assert_eq!(combinations(5, 2), 10);
        assert_eq!(combinations(5, 5), 1);
        assert_eq!(combinations(5, 0), 1);
        assert_eq!(combinations(2, 5), 0);
        assert_eq!(combinations(110, 9), 4_643_330_358_810);
    }

    #[test]
    fn test_generalized_series_sum() {
        assert_eq!(generalized_series_sum(5, 0), 15); // sum of integers
        assert_eq!(generalized_series_sum(5, 1), 55); // sum of squares
        assert_eq!(generalized_series_sum(5, 2), 140);
        assert_eq!(generalized_series_sum(5, 3), 294);
    }

    /// Every family of the reference law table pinned at `n = 5` against
    /// the corrected closed forms.
    #[test]
    fn test_seven_position_law_table() {
        let n = 5;
        // Simplex numbers C(n + d, d + 1) by dimension.
        assert_eq!(simplex_number(n, 1), 15);
        assert_eq!(simplex_number(n, 2), 35);
        assert_eq!(simplex_number(n, 3), 70);
        assert_eq!(simplex_number(n, 4), 126);
        // Iterated sums of squares; the depth-3 closed form is
        // n(n+1)(n+2)(n+3)(2n+3)/120, consistent with the basis form
        // (2n+3)/5 * S(n, 3).
        assert_eq!(iterated_square_sum(n, 1), 55);
        assert_eq!(iterated_square_sum(n, 2), 105);
        assert_eq!(iterated_square_sum(n, 3), 182);
        assert_eq!(
            iterated_square_sum(n, 3) * 120,
            5 * 6 * 7 * 8 * 13 // n(n+1)(n+2)(n+3)(2n+3)
        );
        // Index-weighted sums of simplex numbers.
        assert_eq!(generalized_series_sum(n, 2), 140); // sum k * T(k)
        assert_eq!(generalized_series_sum(n, 3), 294); // sum k * tetrahedral(k)
        assert_eq!(weighted_partial_square_sum(n), 448); // sum k * pyramidal(k)
        // Sum of cubes (Nicomedes) and its double sum.
        assert_eq!(power_sum(n, 3), 225);
        assert_eq!(power_sum(n, 3), simplex_number(n, 1).pow(2));
        assert_eq!(iterated_power_sum(n, 3, 2), 371); // 1 + 9 + 36 + 100 + 225
        // Interleaved cone-pile sum.
        assert_eq!(interleaved_series_sum(n), 42);
    }


    #[test]
    fn test_multiset_coefficient() {
        assert_eq!(multiset_coefficient(0, 0), 1);
        assert_eq!(multiset_coefficient(0, 3), 0);
        assert_eq!(multiset_coefficient(5, 0), 1);
        assert_eq!(multiset_coefficient(5, 1), 5);
        assert_eq!(multiset_coefficient(5, 2), 15);
        assert_eq!(multiset_coefficient(5, 3), 35);
        assert_eq!(multiset_coefficient(1, 100), 1);
    }

    #[test]
    fn test_segment_simplex_sum_vandermonde() {
        assert_eq!(segment_simplex_sum_vandermonde(4, 5, 0), 30);
        assert_eq!(segment_simplex_sum_vandermonde(4, 5, 1), 110);
        assert_eq!(segment_simplex_sum_vandermonde(4, 5, 2), 315);
        assert_eq!(segment_simplex_sum_vandermonde(1, 5, 2), 70);
        assert_eq!(segment_simplex_sum_vandermonde(0, 4, 1), 10);
        assert_eq!(segment_simplex_sum_vandermonde(7, 0, 3), 0);
    }

    #[test]
    fn test_weighted_segment_simplex_sum() {
        assert_eq!(weighted_segment_simplex_sum(2, 3, 1), 31); // 3*3 + 2*6 + 1*10
        assert_eq!(weighted_segment_simplex_sum(1, 4, 0), 20); // 4*1 + 3*2 + 2*3 + 1*4
        assert_eq!(weighted_segment_simplex_sum(0, 3, 1), 5); // 3*0 + 2*1 + 1*3
        assert_eq!(weighted_segment_simplex_sum(5, 0, 2), 0);
    }

    #[test]
    fn test_arithmetic_series_sum() {
        assert_eq!(arithmetic_series_sum(4, 4, 1), 22); // 4 + 5 + 6 + 7
        assert_eq!(arithmetic_series_sum(1, 10, 1), 55); // 1 + 2 + ... + 10
        assert_eq!(arithmetic_series_sum(1, 5, 2), 25); // odd numbers
        assert_eq!(arithmetic_series_sum(7, 0, 3), 0); // empty sequence
    }

    #[test]
    fn test_centered_expansion() {
        assert_eq!(centered_expansion(0, 6), 1); // center point only
        assert_eq!(centered_expansion(2, 6), 19); // hexagonal: 1, 7, 19
        assert_eq!(centered_expansion(2, 8), 25); // odd square (2*2+1)^2
        assert_eq!(centered_expansion(3, 8), 49);
    }

    #[test]
    fn test_iterated_square_sum() {
        assert_eq!(iterated_square_sum(5, 1), 55);
        assert_eq!(iterated_square_sum(5, 2), 105);
        assert_eq!(iterated_square_sum(5, 3), 182);
    }

    #[test]
    fn test_weighted_partial_square_sum() {
        // 1*1 + 2*5 + 3*14 + 4*30 + 5*55 = 448
        assert_eq!(weighted_partial_square_sum(5), 448);
    }

    #[test]
    fn test_weighted_shifted_square_sum() {
        assert_eq!(weighted_shifted_square_sum(1, 5), 448); // m = 1: no shift
        assert_eq!(weighted_shifted_square_sum(4, 5), 1783);
    }

    #[test]
    fn test_weighted_shifted_pyramidal_sum() {
        assert_eq!(weighted_shifted_pyramidal_sum(4, 5), 1993);
        // m = 1: sum k * (1^2 + ... + k^2) == weighted_partial_square_sum
        assert_eq!(weighted_shifted_pyramidal_sum(1, 5), 448);
    }

    #[test]
    fn test_centered_numbers() {
        assert_eq!(centered_triangular_number(3), 19);
        assert_eq!(centered_hexagonal_number(3), 37);
        assert_eq!(odd_square(3), 49);
        // All three are centered_expansion with 3, 6 and 8 sides.
        for n in 0..=10u64 {
            assert_eq!(centered_triangular_number(n), centered_expansion(n, 3));
            assert_eq!(centered_hexagonal_number(n), centered_expansion(n, 6));
            assert_eq!(odd_square(n), centered_expansion(n, 8));
        }
    }

    #[test]
    fn test_cube_sums_via_power_sums() {
        // sum of cubes == power_sum(n, 3) == (n(n+1)/2)^2
        assert_eq!(power_sum(4, 3), 100);
        assert_eq!(iterated_power_sum(4, 3, 2), 146); // 1 + 9 + 36 + 100
    }

    #[test]
    fn test_interleaved_series_sum() {
        // 1 + 3 + 7 + 12 + 19 = 42
        assert_eq!(interleaved_series_sum(5), 42);
        // ... + 27 + 37 + 48 = 154
        assert_eq!(interleaved_series_sum(8), 154);
    }

    #[test]
    fn test_iterated_interleaved_series_sum() {
        // W = 1, 4, 11, 23, 42, ...
        assert_eq!(iterated_interleaved_series_sum(1), 1);
        assert_eq!(iterated_interleaved_series_sum(4), 39); // 1 + 4 + 11 + 23
        assert_eq!(iterated_interleaved_series_sum(5), 81); // + 42
        // Problem 11: a 14-layer pile plus a 13-layer pile cost 5565.
        assert_eq!(
            iterated_interleaved_series_sum(14) + iterated_interleaved_series_sum(13),
            5565,
        );
    }

    #[test]
    fn test_positive_integer_root() {
        use crate::roots::positive_integer_root;

        // n^3 + 30 n^2 + 29 n - 6396 == 0
        assert_eq!(positive_integer_root(&[-6396, 29, 30, 1]), Some(12));
        // (x - 1)(x - 2) = x^2 - 3x + 2: smallest positive root.
        assert_eq!(positive_integer_root(&[2, -3, 1]), Some(1));
        // x^2 + 1 has no integer root.
        assert_eq!(positive_integer_root(&[1, 0, 1]), None);
        // x * (x - 5) = x^2 - 5x: zero constant term.
        assert_eq!(positive_integer_root(&[0, -5, 1]), Some(5));
        // Huge constant term: Newton-Raphson stage, verified exactly.
        // 10^13 * (x - 3)(x - 7) = 10^13 x^2 - 10^14 x + 21 * 10^13.
        assert_eq!(
            positive_integer_root(&[210_000_000_000_000, -100_000_000_000_000, 10_000_000_000_000]),
            Some(7)
        );
    }

    #[test]
    fn test_forward_difference() {
        assert_eq!(forward_difference(3, 10), 7);
    }

    #[test]
    fn test_newton_forward_interpolation() {
        let diffs = [2.0, 0.5, 0.1];
        let value = newton_forward_interpolation(10.0, 0.5, &diffs);
        assert!((value - 10.94375).abs() < 1e-12);
    }

    #[test]
    fn test_iterated_segment_simplex_sum() {
        assert_eq!(iterated_segment_simplex_sum(4, 5, 1, 0), 36); // T(8)
        assert_eq!(iterated_segment_simplex_sum(4, 5, 1, 1), 110);
        assert_eq!(iterated_segment_simplex_sum(2, 3, 1, 2), 31);
        assert_eq!(iterated_segment_simplex_sum(2, 4, 1, 3), 111);
        assert_eq!(
            iterated_segment_simplex_sum(1, 6, 0, 5),
            simplex_number(6, 5)
        );
        assert_eq!(iterated_segment_simplex_sum(3, 0, 2, 2), 0);
    }

    #[test]
    fn test_rectangular_frustum_sum() {
        assert_eq!(rectangular_frustum_sum(1, 1, 5), 55);
        assert_eq!(rectangular_frustum_sum(3, 3, 3), 50);
        assert_eq!(rectangular_frustum_sum(2, 4, 3), 47);
        assert_eq!(rectangular_frustum_sum(0, 0, 4), 14); // 0 + 1 + 4 + 9
        assert_eq!(rectangular_frustum_sum(9, 9, 0), 0);
    }

    #[test]
    fn test_segment_iterated_square_sum() {
        assert_eq!(segment_iterated_square_sum(3, 3, 0), 50);
        assert_eq!(segment_iterated_square_sum(3, 3, 1), 99);
        assert_eq!(
            segment_iterated_square_sum(1, 5, 2),
            iterated_square_sum(5, 3)
        );
        assert_eq!(segment_iterated_square_sum_offset_expansion(3, 3, 0), 50);
        assert_eq!(segment_iterated_square_sum_offset_expansion(3, 3, 1), 99);
        assert_eq!(segment_iterated_square_sum_offset_expansion(0, 4, 1), 20); // 0 + 1 + 5 + 14
    }

    #[test]
    fn test_iterated_segment_square_sum() {
        assert_eq!(iterated_segment_square_sum(3, 3, 0), 25);
        assert_eq!(iterated_segment_square_sum(3, 3, 1), 50);
        assert_eq!(iterated_segment_square_sum(3, 3, 2), 84);
        // n + 3 C(n, 2) + 2 C(n, 3) at n = 6
        assert_eq!(iterated_segment_square_sum(1, 6, 1), 91);
    }

    #[test]
    fn test_newton_iterated_sum() {
        assert_eq!(leading_differences(&[]), Vec::<i128>::new());
        assert_eq!(leading_differences(&[9, 16, 25, 36]), vec![9, 7, 2, 0]);
        assert_eq!(newton_iterated_sum(1, &[3, 2], 5, 1), 55);
        assert_eq!(newton_iterated_sum(9, &[7, 2], 3, 2), 84);
        assert_eq!(newton_iterated_sum(9, &[7, 2], 3, 0), 25);
        assert_eq!(newton_iterated_sum(10, &[-3], 4, 1), 22);
        assert_eq!(newton_iterated_sum(10, &[-3], 0, 1), 0);
    }

    #[test]
    fn test_iterated_segment_iterated_square_sum() {
        assert_eq!(iterated_segment_iterated_square_sum(3, 3, 1, 0), 55);
        assert_eq!(iterated_segment_iterated_square_sum(3, 3, 1, 1), 99);
        assert_eq!(iterated_segment_iterated_square_sum(3, 3, 1, 2), 157);
        assert_eq!(
            iterated_segment_iterated_square_sum_offset_expansion(3, 3, 1, 2),
            157
        );
        assert_eq!(iterated_segment_iterated_square_sum(0, 4, 1, 2), 27); // 4*0 + 3*1 + 2*5 + 1*14
    }

    #[test]
    fn test_index_weighted_triangular_sums() {
        assert_eq!(generalized_series_sum(4, 2), 65);
        assert_eq!(segment_generalized_series_sum(3, 3, 2), 133);
        assert_eq!(ascending_weighted_segment_simplex_sum(3, 3, 1), 71);
        assert_eq!(ascending_weighted_segment_simplex_sum(0, 3, 1), 11); // 1*0 + 2*1 + 3*3
        assert_eq!(trapezoid_weighted_segment_simplex_sum(3, 3, 0), 53);
        assert_eq!(trapezoid_weighted_segment_simplex_sum(1, 4, 0), 65);
    }

    #[test]
    fn test_power_sums() {
        assert_eq!(stirling_second_kind(4, 2), 7);
        assert_eq!(stirling_second_kind(2, 4), 0);
        assert_eq!(power_forward_differences(1, 3), vec![1, 7, 12, 6]);
        assert_eq!(power_forward_differences(0, 0), vec![1]);
        assert_eq!(power_sum(5, 3), 225);
        assert_eq!(power_sum(0, 3), 0);
        assert_eq!(iterated_power_sum(4, 3, 2), 146);
        assert_eq!(iterated_power_sum(4, 3, 0), 64);
        assert_eq!(iterated_segment_power_sum(2, 3, 3, 1), 99);
        assert_eq!(iterated_segment_power_sum(2, 3, 3, 2), 142);
    }

    #[test]
    fn test_simplex_convolution() {
        assert_eq!(simplex_convolution(4, 1, 1), 56);
        assert_eq!(simplex_convolution(5, 0, 2), simplex_number(5, 4));
        assert_eq!(simplex_convolution(0, 3, 3), 0);
    }
}
