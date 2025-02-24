use crate::bspline::basis::Basis;
use crate::bspline::control_points::OControlPoints;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Point, RealField, SVector};
use std::marker::PhantomData;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use crate::bspline::spline_basis::SplineBasis;
use crate::knots::index::MultiIndex;

/// A [`D`]-dimensional B-spline manifold embedded [`M`]-dimensional euclidian space.
#[derive(Debug, Clone)]
pub struct Spline<T, Knt, Idx, S, const M: usize, N>
where
    T: RealField,
    Knt: Copy,
    S: Basis<T, Knt, Idx>,
    N: Dim,
    DefaultAllocator: Allocator<Const<M>, N>
{

    /// Control points for each parametric direction.
    pub control_points: OControlPoints<T, M, N>,

    /// B-spline basis functions for the parametrization.
    pub space: S,

    phantoms: PhantomData<(Knt, Idx)>
}

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<T, const M: usize, N> = Spline<T, T, usize, SplineBasis<T>, M, N>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<T, const M: usize, N> = Spline<T, SVector<T, 2>, MultiIndex<usize, 2>, MultiSplineBasis<T, 2>, M, N>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<T, const M: usize, N> = Spline<T, SVector<T, 3>, MultiIndex<usize, 3>, MultiSplineBasis<T, 3>, M, N>;

impl<T, Knt, Idx, S, const M : usize, N> Spline<T, Knt, Idx, S, M, N>
where
    T: RealField,
    Knt: Copy,
    S: Basis<T, Knt, Idx>,
    N: Dim,
    DefaultAllocator: Allocator<Const<M>, N>
{
    pub fn new(control_points: OControlPoints<T, M, N>, basis: S) -> Option<Self> {
        (basis.num() == control_points.num()).then_some(Spline { control_points, space: basis, phantoms: Default::default() })
    }

    pub fn eval(&self, t: Knt) -> Point<T, M> {
        let span = self.space.find_span(t).unwrap();
        let idx = self.space.nonzero(&span);
        let b = self.space.eval(t, &span);
        let c = self.control_points.get_nonzero(idx);
        Point::from(c.coords * b)
    }
}