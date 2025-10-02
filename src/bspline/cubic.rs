use crate::space::eval::{EvalBasis, EvalGrad};
use crate::space::traits::Basis;
use nalgebra::{matrix, Const, Dyn, OMatrix, RealField, RowDVector, RowSVector, SMatrix, U1};
use std::sync::LazyLock;

/// The `4✕4` basis transformation matrix for [`CubicBspline::Smooth`].
static MAT_SMOOTH: LazyLock<SMatrix<f64, 4, 4>> = LazyLock::new(|| {
    matrix![
        -1, 3, -3, 1;
        3, -6, 3, 0;
        -3, 0, 3, 0;
        1, 4, 1, 0
    ].cast() / 6.0
});

/// The `3✕4` basis transformation matrix for the derivative of [`CubicBspline::Smooth`].
static MAT_SMOOTH_DERIV: LazyLock<SMatrix<f64, 3, 4>> = LazyLock::new(|| {
    matrix![
        -1, 3, -3, 1;
        2, -4, 2, 0;
        -1, 0, 1, 0
    ].cast() / 2.0
});

/// The `3✕3` basis transformation matrix for [`CubicBspline::Interpolating`].
static MAT_INTERPOLATING: LazyLock<SMatrix<f64, 3, 3>> = LazyLock::new(|| {
    matrix![
        1, -2, 1;
        -6, 6, 0;
        6, 0, 0
    ].cast() / 6.0
});

/// The `2✕3` basis transformation matrix for the derivative of [`CubicBspline::Interpolating`].
static MAT_INTERPOLATING_DERIV: LazyLock<SMatrix<f64, 2, 3>> = LazyLock::new(|| {
    matrix![
        1, -2, 1;
        -2, 2, 0
    ].cast() / 2.0
});

/// Cubic B-Spline basis functions on `[0,1]`.
#[derive(Clone, Copy, Debug)]
pub enum CubicBspline {
    /// Smooth basis functions that are `C²` at the boundaries of `[0,1]`.
    Smooth,

    /// Interpolating basis functions that are `C⁰` at the left boundary of `[0,1]`.
    Interpolating
}

impl CubicBspline {
    /// Evaluates the `4` smooth cubic B-Splines at `x`
    /// as the row-vector `(b[1],b[2],b[3],b[4])`.
    pub fn eval_smooth<T: RealField + Copy>(x: T) -> RowSVector<T, 4> {
        let pows = matrix![x.powi(3), x.powi(2), x, T::one()];
        pows * MAT_SMOOTH.cast()
    }

    /// Evaluates the `3` interpolating cubic B-Splines at `x`
    /// as the row-vector `(b[1],b[2],b[3])`.
    pub fn eval_interpolating<T: RealField + Copy>(x: T) -> RowSVector<T, 3> {
        let pows = matrix![x.powi(3), x, T::one()];
        pows * MAT_INTERPOLATING.cast()
    }

    /// Evaluates the derivatives of the `4` smooth cubic B-Splines at `x`
    /// as the row-vector `(b[1]',b[2]',b[3]',b[4]')`.
    pub fn eval_smooth_deriv<T: RealField + Copy>(x: T) -> RowSVector<T, 4> {
        let pows = matrix![x.powi(2), x, T::one()];
        pows * MAT_SMOOTH_DERIV.cast()
    }

    /// Evaluates the derivatives of the `3` interpolating cubic B-Splines at `x`
    /// as the row-vector `(b[1]',b[2]',b[3]')`.
    pub fn eval_interpolating_deriv<T: RealField + Copy>(x: T) -> RowSVector<T, 3> {
        let pows = matrix![x.powi(2), T::one()];
        pows * MAT_INTERPOLATING_DERIV.cast()
    }
}

impl Basis for CubicBspline {
    type NumBasis = Dyn;
    type NumComponents = U1;
    type Coord<T> = T;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CubicBspline::Smooth => Dyn(4),
            CubicBspline::Interpolating => Dyn(3)
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T> for CubicBspline {
    fn eval(&self, x: T) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        match self {
            CubicBspline::Smooth => {
                RowDVector::from_row_slice(CubicBspline::eval_smooth(x).as_slice())
            },
            CubicBspline::Interpolating => {
                RowDVector::from_row_slice(CubicBspline::eval_interpolating(x).as_slice())
            },
        }
    }
}

impl<T: RealField + Copy> EvalGrad<T, 1> for CubicBspline {
    fn eval_grad(&self, x: T) -> OMatrix<T, Const<1>, Self::NumBasis> {
        match self {
            CubicBspline::Smooth => {
                RowDVector::from_row_slice(CubicBspline::eval_smooth_deriv(x).as_slice())
            },
            CubicBspline::Interpolating => {
                RowDVector::from_row_slice(CubicBspline::eval_interpolating_deriv(x).as_slice())
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq};
    use itertools::izip;
    use nalgebra::dvector;
    use rand::random_range;
    use super::*;

    #[test]
    fn eval_smooth() {
        let basis = CubicBspline::Smooth;

        // Test exact values
        let ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let evals_exact = [
            dvector![0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0],
            dvector![8.5333333333333e-02, 6.3066666666667e-01, 2.8266666666667e-01, 1.3333333333333e-03],
            dvector![3.6e-02, 5.38666666666667e-01, 4.14666666666667e-01, 1.06666666666667e-02],
            dvector![1.06666666666667e-02, 4.14666666666667e-01, 5.38666666666667e-01, 3.6e-02],
            dvector![1.3333333333333e-03, 2.82666666666667e-01, 6.30666666666667e-01, 8.53333333333333e-02],
            dvector![0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667],
        ];

        for (t, eval_exact) in izip!(ts, evals_exact) {
            let eval = basis.eval(t).transpose();
            assert_abs_diff_eq!(eval, eval_exact, epsilon = 1e-13);
        }

        // Test if sum equals 1
        let t = random_range(0.0..=1.0);
        let eval_sum = basis.eval(t).sum();
        assert_abs_diff_eq!(eval_sum, 1.0);
    }

    #[test]
    fn eval_interpolating() {
        let basis = CubicBspline::Interpolating;

        // Test if sum equals 1
        let t = random_range(0.0..=1.0);
        let eval_sum = basis.eval(t).sum();
        assert_abs_diff_eq!(eval_sum, 1.0);
    }

    #[test]
    fn eval_smooth_deriv() {
        let basis = CubicBspline::Smooth;

        // Test if sum equals 0
        let t = random_range(0.0..=1.0);
        let eval_sum = basis.eval_grad(t).sum();
        assert_abs_diff_eq!(eval_sum, 0.0);
    }

    #[test]
    fn eval_interpolating_deriv() {
        let basis = CubicBspline::Interpolating;

        // Test if sum equals 0
        let t = random_range(0.0..=1.0);
        let eval_sum = basis.eval_grad(t).sum();
        assert_abs_diff_eq!(eval_sum, 0.0);
    }
}