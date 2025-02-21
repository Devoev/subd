use crate::bspline::control_points::OControlPoints;
use crate::bspline::spline_basis::{MultiSplineBasis, SplineBasis, SplineBasis1};
use crate::knots::knot_vec::KnotVec;
use crate::knots::multi_knot_vec::MultiKnotVec;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Point, RealField};
use crate::knots::knots_trait::Knots;

/// A [`D`]-dimensional B-spline manifold embedded [`M`]-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct Spline<T, K, const D : usize, const M : usize, C> 
where 
    T: RealField,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Control points for each parametric direction.
    pub control_points: OControlPoints<T, M, C>,

    /// B-spline basis functions for the parametrization.
    pub basis: SplineBasis<K>
}

impl<T, const M : usize, C> Spline<T, KnotVec<T>, 1, M, C> 
where 
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Constructs a new [`Spline`].
    pub fn new(control_points: OControlPoints<T, M, C>, basis: SplineBasis1<T>) -> Option<Self> {
        (basis.num() == control_points.num()).then_some(Spline { control_points, basis })
    }
}

impl<T, const D : usize, const M : usize, C> Spline<T, MultiKnotVec<T, D>, D, M, C> 
where 
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C> 
{

    /// Constructs a new [`Spline`].
    pub fn new(control_points: OControlPoints<T, M, C>, basis: MultiSplineBasis<T, D>) -> Option<Self> {
        (basis.num() == control_points.num()).then_some(Spline { control_points, basis })
    }
}

impl<T, const M : usize, C> Spline<T, KnotVec<T>, 1, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline curve at the parametric point `t`.
    pub fn eval_curve(&self, t: T) -> Point<T, M> {
        // todo: remove this method
        let span = self.basis.knots.find_span(t).unwrap();
        let b = self.basis.eval(t);
        let c = self.control_points.get_nonzero(span.nonzero_indices());
        Point::from(c.coords * b)
    }
}

impl<T, const D: usize, const M : usize, C> Spline<T, MultiKnotVec<T, D>, D, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline at the parametric point `t`.
    pub fn eval(&self, t: [T; D]) -> Point<T, M> {
        // todo: debug this function
        let span = self.basis.knots.find_span(t).unwrap();
        let b = self.basis.eval(t);
        let c = self.control_points.get_nonzero(span.nonzero_lin_indices());
        Point::from(c.coords * b)
    }
}