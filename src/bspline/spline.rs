use crate::bspline::control_points::OControlPoints;
use crate::bspline::spline_basis::SplineBasis;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Point, RealField};
use crate::knots::index::Linearize;
use crate::bspline::basis::Basis;

/// A [`D`]-dimensional B-spline manifold embedded [`M`]-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct Spline<T, S, const D : usize, const M : usize, C> 
where 
    T: RealField,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Control points for each parametric direction.
    pub control_points: OControlPoints<T, M, C>,

    /// B-spline basis functions for the parametrization.
    pub space: S
}

impl<T, const M : usize, C> Spline<T, SplineBasis<T>, 1, M, C> 
where 
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Constructs a new [`Spline`].
    pub fn new(control_points: OControlPoints<T, M, C>, basis: SplineBasis<T>) -> Option<Self> {
        (basis.num() == control_points.num()).then_some(Spline { control_points, space: basis })
    }
}

impl<T, const D : usize, const M : usize, C> Spline<T, MultiSplineBasis<T, D>, D, M, C> 
where 
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Constructs a new [`Spline`].
    pub fn new(control_points: OControlPoints<T, M, C>, basis: MultiSplineBasis<T, D>) -> Option<Self> {
        (basis.num() == control_points.num()).then_some(Spline { control_points, space: basis })
    }
}

impl<T, const M : usize, C> Spline<T, SplineBasis<T>, 1, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline curve at the parametric point `t`.
    pub fn eval_curve(&self, t: T) -> Point<T, M> {
        // todo: remove this method
        let span = self.space.find_span(t).unwrap();
        let b = self.space.eval(t);
        let c = self.control_points.get_nonzero(span.nonzero_indices());
        Point::from(c.coords * b)
    }
}

impl<T, const D: usize, const M : usize, C> Spline<T, MultiSplineBasis<T, D>, D, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline at the parametric point `t`.
    pub fn eval(&self, t: [T; D]) -> Point<T, M> {
        // todo: debug this function
        let span = self.space.find_span(t).unwrap();
        let strides = self.space.strides();
        let idx = span.nonzero_indices().linearize(&strides);
        let b = self.space.eval(t);
        let c = self.control_points.get_nonzero(idx);
        Point::from(c.coords * b)
    }
}