use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::traits::Basis;
use nalgebra::{matrix, Const, DVector, Dyn, OMatrix, RealField, RowDVector, SMatrix, U1, U3, U4};
use std::sync::LazyLock;

/// The `4✕4` basis transformation matrix for [`Smooth`].
static MAT_SMOOTH: LazyLock<SMatrix<f64, 4, 4>> = LazyLock::new(|| {
    matrix![
        -1, 3, -3, 1;
        3, -6, 3, 0;
        -3, 0, 3, 0;
        1, 4, 1, 0
    ].cast() / 6.0
});

/// The `3✕4` basis transformation matrix for the derivative of [`Smooth`].
static MAT_SMOOTH_DERIV: LazyLock<SMatrix<f64, 3, 4>> = LazyLock::new(|| {
    matrix![
        -1, 3, -3, 1;
        2, -4, 2, 0;
        -1, 0, 1, 0
    ].cast() / 2.0
});

/// The `3✕3` basis transformation matrix for [`Interpolating`].
static MAT_INTERPOLATING: LazyLock<SMatrix<f64, 3, 3>> = LazyLock::new(|| {
    matrix![
        1, -2, 1;
        -6, 6, 0;
        6, 0, 0
    ].cast() / 6.0
});

/// The `2✕3` basis transformation matrix for the derivative of [`Interpolating`].
static MAT_INTERPOLATING_DERIV: LazyLock<SMatrix<f64, 2, 3>> = LazyLock::new(|| {
    matrix![
        1, -2, 1;
        -2, 2, 0
    ].cast() / 2.0
});

// todo: remove fields from bases below and just cast every time (cost is negligible)

/// Cubic B-Spline basis functions on `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub enum CubicBspline<T> {
    Smooth(Smooth<T>),
    Interpolating(Interpolating<T>)
}

/// Cubic B-spline basis functions that are `C²` at the boundaries of `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub struct Smooth<T>(SMatrix<T, 4, 4>, SMatrix<T, 3, 4>);

/// Cubic B-spline basis functions that are `C⁰` at the left boundary of `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub struct Interpolating<T>(SMatrix<T, 3, 3>, SMatrix<T, 2, 3>);

impl<T: RealField> Smooth<T> {
    /// Constructs a new [`Smooth`].
    pub fn new() -> Self {
        Smooth((*MAT_SMOOTH).cast(), (*MAT_SMOOTH_DERIV).cast())
    }
}

impl<T: RealField> Interpolating<T> {
    /// Constructs a new [`Interpolating`].
    pub fn new() -> Self {
        Interpolating((*MAT_INTERPOLATING).cast(), (*MAT_INTERPOLATING_DERIV).cast())
    }
}

impl <T> Basis for Smooth<T> {
    type NumBasis = U4;
    type NumComponents = U1;

    fn num_basis(&self) -> usize {
        4
    }

    fn num_basis_generic(&self) -> Self::NumBasis {
        U4
    }

    fn num_components(&self) -> usize {
        1
    }

    fn num_components_generic(&self) -> Self::NumComponents {
        U1
    }
}

impl <T: RealField + Copy> EvalBasis<T, T> for Smooth<T> {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let pows = matrix![x.powi(3), x.powi(2), x, T::one()];
        pows * self.0
    }
}

impl<T: RealField + Copy> EvalGrad<T, T, 1> for Smooth<T> {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Self::NumBasis> {
        let pows = matrix![x.powi(2), x, T::one()];
        pows * self.1
    }
}

impl<T> Basis for Interpolating<T> {
    type NumBasis = U3;
    type NumComponents = U1;

    fn num_basis(&self) -> usize {
        3
    }

    fn num_basis_generic(&self) -> Self::NumBasis {
        U3
    }

    fn num_components(&self) -> usize {
        1
    }

    fn num_components_generic(&self) -> Self::NumComponents {
        U1
    }
}

impl <T: RealField + Copy> EvalBasis<T, T> for Interpolating<T> {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let pows = matrix![x.powi(3), x, T::one()];
        pows * self.0
    }
}

impl<T: RealField + Copy> EvalGrad<T, T, 1> for Interpolating<T> {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Self::NumBasis> {
        let pows = matrix![x.powi(2), T::one()];
        pows * self.1
    }
}

impl<T: RealField + Copy> Basis for CubicBspline<T> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis(&self) -> usize {
        match self {
            CubicBspline::Smooth(_) => 4,
            CubicBspline::Interpolating(_) => 3
        }
    }

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.num_basis())
    }

    fn num_components(&self) -> usize {
        1
    }

    fn num_components_generic(&self) -> Self::NumComponents {
        U1
    }
}

impl <T: RealField + Copy> EvalBasis<T, T> for CubicBspline<T> {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        match self {
            CubicBspline::Smooth(basis) => RowDVector::from_row_slice(basis.eval(x).as_slice()),
            CubicBspline::Interpolating(basis) => RowDVector::from_row_slice(basis.eval(x).as_slice()),
        }
    }
}