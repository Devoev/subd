use crate::bspline::basis::BsplineBasis;
use crate::bspline::spline::Spline;
use crate::bspline::de_boor::{DeBoor, MultiDeBoor};
use nalgebra::allocator::Allocator;
use nalgebra::{Dyn, SVector};
use nalgebra::{ComplexField, Const, DefaultAllocator, Dim};

// todo: change to type alias instead of newtype when vector valued splines are implemented

/// A [`D`]-dimensional B-spline geometry embedded [`M`]-dimensional Euclidean space.
#[derive(Debug, Clone)]
pub struct SplineGeo<'a, T, X, B, const M: usize, N>(pub Spline<'a, T, X, B, M, N>)
    where T: ComplexField,
          B: BsplineBasis<T::RealField, X>,
          N: Dim,
          DefaultAllocator: Allocator<Const<M>, N>;

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<'a, T, const M: usize, N = Dyn> = SplineGeo<'a, T, T, DeBoor<T>, M, N>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<'a, T, const M: usize, N = Dyn> = SplineGeo<'a, T, SVector<T, 2>, MultiDeBoor<T, 2>, M, N>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<'a, T, const M: usize, N = Dyn> = SplineGeo<'a, T, SVector<T, 3>, MultiDeBoor<T, 3>, M, N>;