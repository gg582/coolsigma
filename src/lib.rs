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
//!
//! # Panics
//!
//! Functions panic when the mathematical result does not fit in `u128`.
//!
//! # Modules
//!
//! - [`combinations`]: binomial coefficients.
//! - [`sums`]: sums over simplex (binomial-coefficient) sequences.
//! - [`series`]: closed-form sums of polynomial series.
//! - [`interpolation`]: finite differences and Newton forward-difference
//!   interpolation.
//!
//! # Example
//!
//! ```
//! use coolsigma::series::iterated_square_sum;
//! use coolsigma::sums::simplex_sum;
//!
//! assert_eq!(simplex_sum(2, 10), 220); // 10th tetrahedral number
//! assert_eq!(iterated_square_sum(5, 2), 105); // double sum of squares
//! ```

mod common;

pub mod combinations;
pub mod interpolation;
pub mod series;
pub mod sums;

#[cfg(test)]
mod tests {
    use crate::combinations::combinations;
    use crate::interpolation::{forward_difference, newton_forward_interpolation};
    use crate::series::{
        arithmetic_series_sum, centered_expansion, generalized_series_sum, interleaved_series_sum,
        iterated_square_sum, weighted_partial_square_sum,
    };
    use crate::sums::{segment_simplex_sum, simplex_number, simplex_sum, truncated_simplex_sum};

    #[test]
    fn test_simplex_sum() {
        assert_eq!(simplex_sum(0, 0), 0);
        assert_eq!(simplex_sum(1, 10), 55);
        assert_eq!(simplex_sum(2, 10), 220);
        assert_eq!(simplex_sum(3, 10), 715);
        assert_eq!(simplex_sum(100, 10), 4_643_330_358_810);
    }

    #[test]
    fn test_truncated_simplex_sum() {
        assert_eq!(truncated_simplex_sum(4, 1, 6), 15);
        assert_eq!(truncated_simplex_sum(1, 1, 6), 21);
        assert_eq!(truncated_simplex_sum(0, 1, 6), 0);
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

    #[test]
    fn test_simplex_number() {
        assert_eq!(simplex_number(5, 0), 5);
        assert_eq!(simplex_number(5, 1), 15); // triangular
        assert_eq!(simplex_number(5, 2), 35); // tetrahedral
        assert_eq!(simplex_number(5, 3), 70); // pentatope
    }

    #[test]
    fn test_segment_simplex_sum() {
        assert_eq!(segment_simplex_sum(4, 5, 0), 30); // 4 + 5 + 6 + 7 + 8
        assert_eq!(segment_simplex_sum(4, 5, 1), 110);
        assert_eq!(segment_simplex_sum(4, 5, 2), 315);
        assert_eq!(segment_simplex_sum(1, 5, 2), 70); // 1 + 4 + 10 + 20 + 35
        assert_eq!(segment_simplex_sum(0, 4, 1), 10); // 0 + 1 + 3 + 6
        assert_eq!(segment_simplex_sum(7, 0, 3), 0); // empty segment
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
        assert_eq!(centered_expansion(6, 0), 1); // center point only
        assert_eq!(centered_expansion(6, 2), 19); // hexagonal: 1, 7, 19
        assert_eq!(centered_expansion(8, 2), 25); // odd square (2*2+1)^2
        assert_eq!(centered_expansion(8, 3), 49);
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
    fn test_interleaved_series_sum() {
        // 1 + 3 + 7 + 12 + 19 = 42
        assert_eq!(interleaved_series_sum(5), 42);
        // ... + 27 + 37 + 48 = 154
        assert_eq!(interleaved_series_sum(8), 154);
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
}
