use crate::bspline::basis::BsplineBasis;
use crate::bspline::control_points::OControlPoints;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use crate::bspline::space::BsplineSpace;
use crate::bspline::spline_basis::SplineBasis;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, Point, RealField, SVector};

/// A [`D`]-dimensional B-spline geometry embedded [`M`]-dimensional Euclidean space.
#[derive(Debug, Clone)]
pub struct SplineGeo<T, X, const M: usize, B, Nc = Dyn>
where
    T: RealField,
    Nc: Dim,
    B: BsplineBasis<T, X>,
    DefaultAllocator: Allocator<Const<M>, Nc>
{

    /// Control points for each parametric direction.
    pub control_points: OControlPoints<T, M, Nc>,

    /// B-Spline function space for the parametrization.
    pub space: BsplineSpace<T, X, B>
}

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<T, const M: usize, N> = SplineGeo<T, T, M, SplineBasis<T>, N>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<T, const M: usize, N> = SplineGeo<T, SVector<T, 2>, M, MultiSplineBasis<T, 2>, N>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<T, const M: usize, N> = SplineGeo<T, SVector<T, 3>, M, MultiSplineBasis<T, 3>, N>;

impl<T, X, const M: usize, B, Nc> SplineGeo<T, X, M, B, Nc>
where
    T: RealField,
    Nc: Dim,
    B: BsplineBasis<T, X>,
    DefaultAllocator: Allocator<Const<M>, Nc>,
{
    pub fn new(control_points: OControlPoints<T, M, Nc>, space: BsplineSpace<T, X, B>) -> Option<Self> {
        (space.dim() == control_points.num()).then_some(SplineGeo { control_points, space })
    }

    pub fn eval(&self, x: X) -> Point<T, M> {
        let (b, idx) = self.space.basis.eval_nonzero(x);
        let c = self.control_points.get_nonzero(idx);
        Point::from(c.coords * b)
    }
}