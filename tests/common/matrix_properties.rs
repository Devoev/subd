use std::error::Error;
use std::ops::AddAssign;
use approx::{relative_eq, RelativeEq};
use itertools::izip;
use nalgebra::{DMatrix, RealField, Scalar};
use nalgebra_sparse::CooMatrix;
use num_traits::Zero;

/// Asserts that the given `matrix` is symmetric within the given tolerance `epsilon`.
pub fn assert_is_symmetric<T: Scalar + RelativeEq>(matrix: &DMatrix<T>, epsilon: T::Epsilon) where T::Epsilon: Clone {
    assert!(relative_eq!(*matrix, matrix.transpose(), epsilon = epsilon), "Mass matrix is not symmetric");
}

/// Asserts that the given `matrix` is positive definite.
/// Positive definiteness is checked by computing the eigenvalues of the matrix.
/// If the (real-valued) eigenvalues can't be computed, the assertion fails.
pub fn assert_is_positive_definite<T: RealField>(matrix: &DMatrix<T>) -> Result<(), Box<dyn Error>> {
    let eigenvalues = matrix.eigenvalues().ok_or("Failed to compute eigenvalues of matrix")?;
    assert!(eigenvalues.iter().all(|e| e.is_positive()), "matrix is not positive-definite. Some eigenvalues are non-positive.");
    Ok(())
}