use std::iter::zip;
use nalgebra::{Isometry, Point, RealField};
use crate::bspline::spline_basis::SplineBasis;

/// A B-spline curve embedded in `D`-dimensional euclidian space.
pub struct SplineCurve<T : RealField, const D : usize> {

    /// Control points of the curve.
    pub control_points: Vec<Point<T, D>>,

    /// Spline basis of the curve.
    pub basis: SplineBasis<T>
}

impl<T : RealField, const D : usize> SplineCurve<T, D> {

    /// Constructs a new `SplineCurve`.
    pub fn new(control_points: Vec<Point<T, D>>, basis: SplineBasis<T>) -> Self {
        SplineCurve { control_points, basis }
    }
}

impl<T : RealField + Copy, const D : usize> SplineCurve<T, D> {

    pub fn eval(&self, t: T) -> Point<T, D> {
        let idx = self.basis.find_span(t).unwrap();
        let b = self.basis.eval(t);
        let c = &self.control_points[idx - self.basis.p..=idx];
        zip(b, c).fold(Point::<T, D>::origin(), |pos, (bi, ci)| pos + ci.coords * bi)
    }
}