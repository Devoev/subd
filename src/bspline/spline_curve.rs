use std::iter::zip;
use iter_num_tools::lin_space;
use nalgebra::{point, Isometry, Point, RealField};
use crate::bspline::spline_basis::SplineBasis;

/// A B-spline curve embedded in `D`-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct SplineCurve<T : RealField, const D : usize> {

    /// Control points of the curve.
    pub control_points: Vec<Point<T, D>>,

    /// Spline basis of the curve.
    pub basis: SplineBasis<T>
}

impl<T : RealField, const D : usize> SplineCurve<T, D> {

    /// Constructs a new `SplineCurve`.
    pub fn new(control_points: Vec<Point<T, D>>, basis: SplineBasis<T>) -> Option<Self> {
        if control_points.len() == basis.n { Some(SplineCurve { control_points, basis }) }
        else { None }
    }
}

impl<T : RealField + Copy, const D : usize> SplineCurve<T, D> {
    
    /// Evaluates the spline curve at the parametric point `t`.
    pub fn eval(&self, t: T) -> Point<T, D> {
        let idx = self.basis.find_span(t).unwrap();
        let b = self.basis.eval(t);
        let c = &self.control_points[idx - self.basis.p..=idx];
        zip(b, c).fold(Point::origin(), |pos, (bi, ci)| pos + ci.coords * bi)
    }

    /// Meshes this curve by linearly spacing the parametric domain with `num` steps.
    pub fn mesh(&self, num: usize) -> impl Iterator<Item = Point<T, D>> + '_ {
        lin_space(T::zero()..=T::one(), num).map(|t| self.eval(t))
    }
}