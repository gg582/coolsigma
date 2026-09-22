use coolsigma::combinations::combinations;
use coolsigma::series::{arithmetic_series_sum, iterated_square_sum};
use coolsigma::sums::simplex_sum;

fn main() {
    assert_eq!(combinations(10, 3), 120);
    assert_eq!(simplex_sum(2, 4), 20);
    assert_eq!(iterated_square_sum(4, 2), 50);
    assert_eq!(arithmetic_series_sum(5, 4, 1), 26);
    assert_eq!(arithmetic_series_sum(1, 4, 2), 16);
    println!("README example OK");
}
