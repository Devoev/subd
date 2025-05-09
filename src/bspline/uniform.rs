//! Spline bases for uniform knot vectors.

use crate::bspline::basis::BsplineBasis;
use nalgebra::{matrix, vector, Const, DVector, RealField, SMatrix, SVector};
use std::ops::Range;

/// B-Spline basis of dimension [`N`],
/// evaluated by transformation of [`M`] monomial basis `{xᵐ}` functions
/// using a basis transformation matrix.
pub struct MonomialTransform<T, const N: usize, const M: usize> {
    /// The `N✕M` basis transformation matrix.
    basis_mat: SMatrix<T, N, M>,

    /// The `M` powers of the monomial functions.
    monomial_pows: SVector<i32, M>
}

impl <T: RealField + Copy, const N: usize, const M: usize> MonomialTransform<T, N, M> {

    /// Evaluates the basis functions of this basis at the parametric point `x`.
    pub fn eval(&self, x: T) -> SVector<T, N> {
        let x_pow = self.monomial_pows.map(|p| x.powi(p));
        self.basis_mat * x_pow
    }
}

impl <T: RealField + Copy, const N: usize, const M: usize> BsplineBasis<T, T> for MonomialTransform<T, N, M> {
    type NonzeroIndices = Range<usize>;

    fn len(&self) -> usize {
        N
    }

    fn eval_nonzero(&self, x: T) -> (DVector<T>, Self::NonzeroIndices) {
        (DVector::from_column_slice(self.eval(x).as_slice()), 0..N)
    }
}

/// Cubic B-Spline basis on the uniform knot vector `(0,1/7,2/7,3/7,4/7,5/7,6/7,1)`.
/// The basis functions are `C²` at the boundary.
///
/// The basis evaluation is represented by the basis transformation matrix,
/// mapping the monomial basis `{x³,x²,x,1}` to the B-Spline basis.
pub type Uniform3<T> = MonomialTransform<T, 4, 4>;

impl <T: RealField + Copy> Uniform3<T> {
    /// Constructs a new [`Uniform3`].
    pub fn new() -> Self {
        let basis_mat = matrix![
            -1.0, 3.0, -3.0, 1.0;
            3.0, -6.0, 0.0, 4.0;
            -3.0, 3.0, 3.0, 1.0;
            1.0, 0.0, 0.0, 0.0;
        ] / 6.0;

        Self { basis_mat: basis_mat.cast(), monomial_pows: vector![3, 2, 1, 0] }
    }
}

/// Derivatives of the cubic B-Splines [`Uniform3`].
pub type Uniform3Deriv<T> = MonomialTransform<T, 4, 3>;

impl <T: RealField + Copy> Uniform3Deriv<T> {
    /// Constructs a new [`Uniform3Deriv`].
    pub fn new() -> Self {
        let basis_mat = matrix![
            -1.0, 2.0, -1.0;
            3.0, -4.0, 0.0;
            -3.0, 2.0, 1.0;
            1.0, 0.0, 0.0;
        ] / 2.0;

        Self { basis_mat: basis_mat.cast(), monomial_pows: vector![2, 1, 0] }
    }
}

/// Cubic B-Splines on the uniform knot vector which are interpolating at the left boundary `x = 0`.
pub type Uniform3Interp<T> = MonomialTransform<T, 3, 3>;

impl <T: RealField + Copy> Uniform3Interp<T> {
    /// Constructs a new [`Uniform3Interp`].
    pub fn new() -> Self {
        let basis_mat = matrix![
            1.0, -6.0, 6.0;
            -2.0, 6.0, 0.0;
            1.0, 0.0, 0.0
        ] / 6.0;

        Self { basis_mat: basis_mat.cast(), monomial_pows: vector![3, 1, 0] }
    }
}

/// Derivatives of the interpolating cubic B-Splines [`Uniform3Interp`].
pub type Uniform3InterpDeriv<T> = MonomialTransform<T, 3, 2>;

impl <T: RealField + Copy> Uniform3InterpDeriv<T> {
    /// Constructs a new [`Uniform3InterpDeriv`].
    pub fn new() -> Self {
        let basis_mat = matrix![
            1.0, -2.0;
            -2.0, 2.0;
            1.0, 0.0
        ] / 2.0;

        Self { basis_mat: basis_mat.cast(), monomial_pows: vector![2, 0] }
    }
}