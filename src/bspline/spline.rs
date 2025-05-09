use crate::bspline::basis::BsplineBasis;
use crate::bspline::control_points::OControlPoints;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use crate::bspline::spline_basis::SplineBasis;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, Point, RealField, SVector};
use std::marker::PhantomData;

/// A [`D`]-dimensional B-spline manifold embedded [`M`]-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct Spline<T, X, const M: usize, S, Nc = Dyn>
where
    T: RealField,
    Nc: Dim,
    S: BsplineBasis<T, X>,
    DefaultAllocator: Allocator<Const<M>, Nc>
{

    /// Control points for each parametric direction.
    pub control_points: OControlPoints<T, M, Nc>,

    /// B-spline basis functions for the parametrization.
    pub space: S,

    phantoms: PhantomData<X>
}

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<T, const M: usize, N> = Spline<T, T, M, SplineBasis<T>, N>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<T, const M: usize, N> = Spline<T, SVector<T, 2>, M, MultiSplineBasis<T, 2>, N>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<T, const M: usize, N> = Spline<T, SVector<T, 3>, M, MultiSplineBasis<T, 3>, N>;

impl<T, X, const M: usize, S, Nc> Spline<T, X, M, S, Nc>
where
    T: RealField,
    Nc: Dim,
    S: BsplineBasis<T, X>,
    DefaultAllocator: Allocator<Const<M>, Nc>,
{
    pub fn new(control_points: OControlPoints<T, M, Nc>, basis: S) -> Option<Self> {
        (basis.len() == control_points.num()).then_some(Spline { control_points, space: basis, phantoms: Default::default() })
    }

    pub fn eval(&self, x: X) -> Point<T, M> {
        let (b, idx) = self.space.eval_nonzero(x);
        let c = self.control_points.get_nonzero(idx);
        Point::from(c.coords * b)
    }
}