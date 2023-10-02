
use ndarray::{Array2, Array1};

fn main() {
    // Create dummy data for sources and sqrt_weights
    let sources = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let sqrt_weights = Array1::from(vec![2.0, 3.0, 4.0]);

    // Calculate weighted_sources
    let weighted_sources = &sources * &sqrt_weights.t();

    // Print the result
    println!("Weighted Sources:\n{:?}", weighted_sources);
}
