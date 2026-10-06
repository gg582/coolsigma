use coolsigma::combinations::{combinations, multiset_coefficient};
use coolsigma::series::{arithmetic_series_sum, iterated_square_sum};
use coolsigma::sums::{segment_simplex_sum_vandermonde, simplex_sum, weighted_segment_simplex_sum};

fn main() {
    assert_eq!(combinations(10, 3), 120);
    assert_eq!(multiset_coefficient(4, 3), 20);
    assert_eq!(simplex_sum(2, 4), 20);
    assert_eq!(iterated_square_sum(4, 2), 50);
    assert_eq!(arithmetic_series_sum(5, 4, 1), 26);
    assert_eq!(arithmetic_series_sum(1, 4, 2), 16);
    assert_eq!(segment_simplex_sum_vandermonde(4, 5, 1), 110);
    assert_eq!(weighted_segment_simplex_sum(2, 3, 1), 31);
    println!("README example OK");
}
