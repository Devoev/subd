use nalgebra::{Point, RealField};
use crate::bspline::spline_basis::SplineBasis;

/// A B-spline curve embedded in `D`-dimensional euclidian space.
pub struct SplineCurve<T : RealField, const D : usize> {

    /// Control points of the curve.
    pub control_points: Vec<Point<T, D>>,

    /// Spline basis of the curve.
    pub basis: SplineBasis<T>
}