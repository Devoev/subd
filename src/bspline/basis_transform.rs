//! Spline bases for uniform knot vectors.

use crate::basis::traits::{Basis, NumBasis};
use nalgebra::{matrix, vector, Const, Dyn, OMatrix, RealField, RowDVector, SMatrix, SVector};

/// B-Spline basis of dimension [`N`],
/// evaluated by transformation of [`M`] monomial basis functions `{xᵐ}`
/// using a basis transformation matrix.
pub struct BasisTransform<T, const N: usize, const M: usize> {
    /// The `N✕M` basis transformation matrix.
    basis_mat: SMatrix<T, N, M>,

    /// The `M` powers of the monomial functions.
    monomial_pows: SVector<i32, M>
}

impl <T: RealField + Copy, const N: usize, const M: usize> BasisTransform<T, N, M> {
    /// Evaluates the basis functions of this basis at the parametric point `x`.
    pub fn eval(&self, x: T) -> SVector<T, N> {
        let x_pow = self.monomial_pows.map(|p| x.powi(p));
        self.basis_mat * x_pow
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> NumBasis for BasisTransform<T, N, M> {
    fn num_basis(&self) -> usize {
        N
    }
}

impl <T: RealField + Copy, const N: usize, const M: usize> Basis<T, T, 1> for BasisTransform<T, N, M> {
    fn eval(&self, x: T) -> OMatrix<T, Const<1>, Dyn> {
        RowDVector::from_row_slice(self.eval(x).as_slice())
    }
}

// todo: docs are incorrect, because these (probably) are not the correct knot values

/// Cubic B-Spline basis on the uniform knot vector `(0,1/7,2/7,3/7,4/7,5/7,6/7,1)`.
/// The basis functions are `C²` at the boundary.
///
/// The basis evaluation is represented by the basis transformation matrix,
/// mapping the monomial basis `{x³,x²,x,1}` to the B-Spline basis.
pub type Uniform3<T> = BasisTransform<T, 4, 4>;

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
pub type Uniform3Deriv<T> = BasisTransform<T, 4, 3>;

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
pub type Uniform3Interp<T> = BasisTransform<T, 3, 3>;

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
pub type Uniform3InterpDeriv<T> = BasisTransform<T, 3, 2>;

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