use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::traits::Basis;
use nalgebra::{matrix, Const, Dyn, OMatrix, RealField, RowDVector, SMatrix, U1, U3, U4};
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
pub enum CubicBspline {
    Smooth,
    Interpolating
}

/// Cubic B-spline basis functions that are `C²` at the boundaries of `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub struct Smooth;

/// Cubic B-spline basis functions that are `C⁰` at the left boundary of `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub struct Interpolating;

impl Basis for Smooth {
    type NumBasis = U4;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        U4
    }
}

impl <T: RealField + Copy> EvalBasis<T, T> for Smooth {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let pows = matrix![x.powi(3), x.powi(2), x, T::one()];
        pows * MAT_SMOOTH.cast()
    }
}

impl<T: RealField + Copy> EvalGrad<T, T, 1> for Smooth {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Self::NumBasis> {
        let pows = matrix![x.powi(2), x, T::one()];
        pows * MAT_SMOOTH_DERIV.cast()
    }
}

impl Basis for Interpolating {
    type NumBasis = U3;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        U3
    }
}

impl <T: RealField + Copy> EvalBasis<T, T> for Interpolating {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let pows = matrix![x.powi(3), x, T::one()];
        pows * MAT_INTERPOLATING.cast()
    }
}

impl<T: RealField + Copy> EvalGrad<T, T, 1> for Interpolating {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Self::NumBasis> {
        let pows = matrix![x.powi(2), T::one()];
        pows * MAT_INTERPOLATING_DERIV.cast()
    }
}

impl Basis for CubicBspline {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CubicBspline::Smooth => Dyn(4),
            CubicBspline::Interpolating => Dyn(3)
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T, T> for CubicBspline {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        match self {
            CubicBspline::Smooth => RowDVector::from_row_slice(Smooth.eval(x).as_slice()),
            CubicBspline::Interpolating => RowDVector::from_row_slice(Interpolating.eval(x).as_slice()),
        }
    }
}

impl<T: RealField + Copy> EvalGrad<T, T, 1> for CubicBspline {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Self::NumBasis> {
        match self {
            CubicBspline::Smooth => RowDVector::from_row_slice(Smooth.eval_grad(x).as_slice()),
            CubicBspline::Interpolating => RowDVector::from_row_slice(Interpolating.eval_grad(x).as_slice()),
        }
    }
}