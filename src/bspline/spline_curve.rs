use crate::bspline::spline_basis::SplineBasis;
use crate::knots::knot_vec::KnotVec;
use crate::mesh::univariate_mesh::UnivariateMesh;
use nalgebra::{Matrix, Point, RealField};
use std::iter::zip;

/// A B-spline curve embedded in `D`-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct SplineCurve<T : RealField, const D : usize> {

    /// Control points of the curve.
    pub control_points: Vec<Point<T, D>>, // todo: replace with matrix?

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
        let span = self.basis.knots.find_span(t).unwrap();
        let b = self.basis.eval(t);
        let c = &self.control_points[span.nonzero_indices()];
        // todo: implement this using matrix-vector product
        zip(&b, c).fold(Point::origin(), |pos, (bi, ci)| pos + ci.coords * *bi)
    }

    /// Meshes this curve by linearly spacing the parametric domain with `num` steps.
    pub fn mesh(self, num: usize) -> UnivariateMesh<T, D> {
        UnivariateMesh::new(self, KnotVec::uniform(num, 1))
    }
}