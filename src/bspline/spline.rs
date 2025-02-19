use crate::bspline::control_points::OControlPoints;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use crate::knots::multi_knot_vec::MultiKnotVec;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Point, RealField};

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
    pub basis: MultiSplineBasis<T, D>
}

impl<T, const D : usize, const M : usize, C> Spline<T, D, M, C> 
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

impl<T, const M : usize, C> Spline<T, 1, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline curve at the parametric point `t`.
    pub fn eval_curve(&self, t: T) -> Point<T, M> {
        let basis = &self.basis.univariate_bases[0];
        let knots = MultiKnotVec::new([basis.clone().knots]);
        let span = knots.find_span([t], [basis.n]).unwrap();
        let b = basis.eval(t);
        let c = self.control_points.get_nonzero(span, [basis.n], [basis.p]);
        Point::from(c.coords * b)
    }
}

impl<T, const D: usize, const M : usize, C> Spline<T, D, M, C>
where
    T: RealField + Copy,
    C: Dim,
    DefaultAllocator: Allocator<Const<M>, C>
{

    /// Evaluates the spline at the parametric point `t`.
    pub fn eval(&self, t: [T; D]) -> Point<T, M> {
        // todo: debug this function
        let basis = &self.basis;
        let knots_uni = basis.univariate_bases.iter().map(|b| b.knots.clone()).collect_array::<D>().unwrap();
        let knots = MultiKnotVec::new(knots_uni);
        let n: [usize; D] = basis.n().collect_array().unwrap();
        let p: [usize; D] = basis.p().collect_array().unwrap();
        let span = knots.find_span(t, n).unwrap();
        let b = basis.eval(t);
        let c = self.control_points.get_nonzero(span, n, p);
        Point::from(c.coords * b)
    }
}