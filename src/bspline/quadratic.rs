use nalgebra::{matrix, Matrix3, RealField, RowSVector, RowVector3, RowVector4};
use std::sync::LazyLock;

/// The `3✕3` basis transformation matrix for [`QuadraticBspline::Smooth`].
static MAT_SMOOTH: LazyLock<Matrix3<f64>> = LazyLock::new(|| {
    matrix![
        1, 1, 0;
        -2, 2, 0;
        1, -2, 1;
    ].cast() / 2.0
});

/// Uniform quadratic B-Spline basis function on `[0,1]`.
#[derive(Debug, Copy, Clone)]
pub enum QuadraticBspline {
    /// Smooth basis functions that are `C¹` at the boundaries of `[0,1]`.
    Smooth,

    /// Interpolating basis functions that are `C⁰` at the left boundary of `[0,1]`.
    Interpolating
}

impl QuadraticBspline {
    /// Evaluates the `3` smooth quadratic B-Splines at `x`
    /// as the row-vector `(b[1],b[2],b[3])`.
    pub fn eval_smooth<T: RealField + Copy>(x: T) -> RowVector3<T> {
        let pows = matrix![x.powi(2), x, T::one()];
        pows * MAT_SMOOTH.cast()
    }
}