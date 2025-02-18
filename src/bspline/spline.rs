use crate::bspline::control_points::OControlPoints;
use crate::bspline::multivariate_spline_basis::MultivariateSplineBasis;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, Point, RealField};
use nalgebra::allocator::Allocator;

/// A `D`-dimensional B-spline manifold embedded `M`-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct Spline<T, const D : usize, const M : usize, C> 
where 
    T: RealField,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Control points for each parametric direction.
    pub control_points: OControlPoints<T, M, C>,

    /// B-spline basis functions for the parametrization.
    pub basis: MultivariateSplineBasis<T, D>
}

impl<T, const D : usize, const M : usize, C> Spline<T, D, M, C> 
where 
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Constructs a new [`Spline`].
    pub fn new(control_points: OControlPoints<T, M, C>, basis: MultivariateSplineBasis<T, D>) -> Option<Self> {
        (basis.num() == control_points.num()).then_some(Spline { control_points, basis })
    }
}

impl<T, const M : usize, C> Spline<T, 1, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline curve at the parametric point `t`.
    pub fn eval(&self, t: T) -> Point<T, M> {
        let basis = &self.basis.univariate_bases[0];
        let span = basis.knots.find_span(t, basis.n).unwrap();
        let b = basis.eval(t);
        let c = &self.control_points.get_nonzero([span], basis.p);
        Point::from(c.coords * b)
    }
}

impl<T, const M : usize, C> Spline<T, 2, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline at the parametric point `t`.
    pub fn eval(&self, t: [T; 2]) -> Point<T, M> {
        // let idx = self.basis.find_span(t).unwrap();
        // let b = self.basis.eval(t);
        // let c = &self.control_points[idx - self.basis.p..=idx];
        // zip(&b, c).fold(Point::origin(), |pos, (bi, ci)| pos + ci.coords * *bi)
        todo!()
    }
}