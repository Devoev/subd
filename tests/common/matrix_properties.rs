use std::error::Error;
use std::ops::AddAssign;
use approx::{relative_eq, RelativeEq};
use itertools::izip;
use nalgebra::{ComplexField, DMatrix, Matrix, RealField, Scalar};
use nalgebra_sparse::CooMatrix;
use num_traits::Zero;

/// Asserts that the given `matrix` is symmetric within the given tolerance `epsilon`.
pub fn assert_is_symmetric<T: Scalar + RelativeEq>(matrix: &DMatrix<T>, epsilon: T::Epsilon) where T::Epsilon: Clone {
    assert!(relative_eq!(*matrix, matrix.transpose(), epsilon = epsilon), "matrix is not symmetric");
}

/// Asserts that the given `matrix` is of specified `rank` within the given tolerance `epsilon`.
pub fn assert_has_rank<T: ComplexField>(matrix: &DMatrix<T>, rank: usize, epsilon: T::RealField) {
    assert_eq!(matrix.rank(epsilon), rank, "matrix has not rank {rank}");
}

/// Asserts that the given `matrix` is positive definite within the given tolerance `epsilon`.
///
/// Positive definiteness is checked by computing the eigenvalues of the matrix
/// and testing if they are larger than the tolerance.
/// If the (real-valued) eigenvalues can't be computed, the assertion fails.
pub fn assert_is_positive_definite<T: RealField>(matrix: &DMatrix<T>, epsilon: T) -> Result<(), Box<dyn Error>> {
    let eigenvalues = matrix.eigenvalues().ok_or("Failed to compute eigenvalues of matrix")?;
    assert!(eigenvalues.iter().all(|e| *e > epsilon), "matrix is not positive-definite. Some eigenvalues are non-positive.");
    Ok(())
}