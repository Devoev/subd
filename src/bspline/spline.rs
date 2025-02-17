use crate::bspline::multivariate_spline_basis::MultivariateSplineBasis;
use nalgebra::{Const, Dyn, OMatrix, Point, RealField};
use std::iter::zip;

/// Control points of a spline. Stored column wise as a matrix.
type ControlPoints<T, const M : usize> = OMatrix<T, Const<M>, Dyn>;

/// A `D`-dimensional B-spline manifold embedded `M`-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct Spline<T : RealField, const D : usize, const M : usize>  {

    /// Control points for each parametric direction.
    pub control_points: ControlPoints<T, M>,

    /// B-spline basis functions for the parametrization.
    pub basis: MultivariateSplineBasis<T, D>
}

impl<T : RealField + Copy, const D : usize, const M : usize> Spline<T, D, M> {

    /// Returns `true` if there are the same amount of control points as basis functions.
    fn coeffs_match_basis(control_points: &ControlPoints<T, M>, basis: &MultivariateSplineBasis<T, D>) -> bool {
        let (_, num) = control_points.shape();
        basis.num() == num
    }

    /// Constructs a new [`Spline`].
    pub fn new(control_points: ControlPoints<T, M>, basis: MultivariateSplineBasis<T, D>) -> Option<Self> {
        if Self::coeffs_match_basis(&control_points, &basis) {
            Some(Spline { control_points, basis })
        }
        else { None }
    }
}

impl<T : RealField + Copy, const M : usize> Spline<T, 1, M> {

    /// Evaluates the spline curve at the parametric point `t`.
    pub fn eval(&self, t: T) -> Point<T, M> {
        let basis = &self.basis.univariate_bases[0];
        let span = basis.knots.find_span(t, basis.n).unwrap();
        let b = basis.eval(t);
        let c = &self.control_points.columns_range(span.nonzero_indices(basis.p));
        Point::from(c * b)
    }
}

impl<T : RealField + Copy, const M : usize> Spline<T, 2, M> {

    /// Evaluates the spline at the parametric point `t`.
    pub fn eval(&self, t: [T; 2]) -> Point<T, M> {
        // let idx = self.basis.find_span(t).unwrap();
        // let b = self.basis.eval(t);
        // let c = &self.control_points[idx - self.basis.p..=idx];
        // zip(&b, c).fold(Point::origin(), |pos, (bi, ci)| pos + ci.coords * *bi)
        todo!()
    }
}