use crate::bspline::spline_curve::SplineCurve;
use crate::knots::knot_vec::KnotVec;
use nalgebra::{Point, RealField};

/// A one-dimensional Bezier mesh of a [`SplineCurve`].
#[derive(Debug, Clone)]
pub struct UnivariateMesh<T : RealField, const D : usize> {
    /// The spline curve parametrizing the domain.
    curve: SplineCurve<T, D>,
    /// The parametric Bezier mesh.
    parametric_mesh: KnotVec<T>,
}

impl<T: RealField, const D: usize> UnivariateMesh<T, D> {

    /// Constructs a new [`UnivariateMesh`].
    pub fn new(curve: SplineCurve<T, D>, parametric_mesh: KnotVec<T>) -> Self {
        UnivariateMesh { curve, parametric_mesh }
    }
}

impl<T : RealField + Copy, const D : usize> IntoIterator for &UnivariateMesh<T, D> {
    type Item = Point<T, D>;
    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.parametric_mesh.breaks().map(|t| self.curve.eval(t))
    }
}