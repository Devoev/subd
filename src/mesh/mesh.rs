use crate::bspline::spline_curve::SplineCurve;
use crate::knots::knot_vec::KnotVec;
use nalgebra::{Point, RealField};

/// A one-dimensional Bezier mesh of a [`SplineCurve`].
#[derive(Debug, Clone)]
pub struct Mesh<T : RealField, const D : usize> {
    /// The spline curve parametrizing the domain.
    curve: SplineCurve<T, D>,
    /// The parametric Bezier mesh.
    parametric_mesh: KnotVec<T>,
}

impl<T: RealField, const D: usize> Mesh<T, D> {

    /// Constructs a new [`Mesh`].
    pub fn new(curve: SplineCurve<T, D>, parametric_mesh: KnotVec<T>) -> Self {
        Mesh { curve, parametric_mesh }
    }
}

impl<T : RealField + Copy, const D : usize> IntoIterator for Mesh<T, D> {
    type Item = Point<T, D>;
    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.parametric_mesh.breaks().into_iter().map(move |t| self.curve.eval(t))
    }
}