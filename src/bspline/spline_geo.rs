use crate::bspline::de_boor::{DeBoor, DeBoorMulti};
use crate::bspline::spline::Spline;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, Point, RealField, SVector};
use nalgebra::allocator::Allocator;
use crate::bspline::basis::BsplineBasis;
use crate::cells::chart::Chart;
// todo: possibly change back to newtype or seperate type alltogether?

/// A B-spline geometry embedded [`M`]-dimensional Euclidean space.
/// Each spline geometry is a regular [Spline] where each of the [`M`] components is represented
/// by the same basis [`B`].
/// This is equivalent to using points of size [`M`] for each coefficient 
/// and a single scalar valued basis.
pub type SplineGeo<'a, T: RealField, X, B, const M: usize, Nc = Dyn> = Spline<'a, T, X, B, M, 1, Nc>;

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<'a, T, const M: usize, Nc = Dyn> = SplineGeo<'a, T, T, DeBoor<T>, M, Nc>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<'a, T, const M: usize, Nc = Dyn> = SplineGeo<'a, T, SVector<T, 2>, DeBoorMulti<T, 2>, M, Nc>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<'a, T, const M: usize, Nc = Dyn> = SplineGeo<'a, T, SVector<T, 3>, DeBoorMulti<T, 3>, M, Nc>;

impl <'a, T, X, B, const M: usize, Nc> Chart<T, X, M> for SplineGeo<'a, T, X, B, M, Nc>
    where T: RealField,
          B: BsplineBasis<T::RealField, X, 1>,
          Nc: Dim,
          DefaultAllocator: Allocator<Const<M>, Nc>
{
    fn eval(&self, x: X) -> Point<T, M> {
        Point::from(self.eval(x))
    }
}

impl <'a, T, X, B, const M: usize, Nc> Chart<T, X, M> for &'a SplineGeo<'a, T, X, B, M, Nc>
    where T: RealField,
          B: BsplineBasis<T::RealField, X, 1>,
          Nc: Dim,
          DefaultAllocator: Allocator<Const<M>, Nc>
{
    fn eval(&self, x: X) -> Point<T, M> {
        Point::from((*self).eval(x))
    }
}