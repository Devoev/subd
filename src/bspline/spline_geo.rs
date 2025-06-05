use crate::bspline::basis::BsplineBasis;
use crate::bspline::de_boor::{DeBoor, DeBoorMulti};
use crate::bspline::spline::Spline;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{AreMultipliable, ShapeConstraint};
use nalgebra::{ArrayStorage, Const, DefaultAllocator, Dim, Dyn, Matrix, Point, RealField, SMatrix, SVector, U1};
use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
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

impl <T, X, B, const D: usize, const M: usize, Nc> Chart<T, X, D, M> for SplineGeo<'_, T, X, B, M, Nc>
    where T: RealField,
          X: Dimensioned<T, D>,
          B: BsplineBasis<T::RealField, X, 1>,
          Nc: Dim,
          DefaultAllocator: Allocator<Const<M>, Nc>
{
    fn eval(&self, x: X) -> Point<T, M> {
        Point::from(self.eval(x))
    }

    fn eval_diff(&self, x: X) -> SMatrix<T, M, D> {
        todo!("Implement this if BsplineBasis trait is fully replaced with Basis")
    }
}

impl <'a, T, X, B, const D: usize, const M: usize, Nc> Chart<T, X, D, M> for &'a SplineGeo<'a, T, X, B, M, Nc>
    where T: RealField,
          X: Dimensioned<T, D>,
          B: BsplineBasis<T::RealField, X, 1>,
          Nc: Dim,
          DefaultAllocator: Allocator<Const<M>, Nc>
{
    fn eval(&self, x: X) -> Point<T, M> {
        Point::from((*self).eval(x))
    }

    fn eval_diff(&self, x: X) -> SMatrix<T, M, D> {
        todo!("Implement this if BsplineBasis trait is fully replaced with Basis")
    }
}

/// Jacobian matrix of a [`SplineGeo`].
pub struct Jacobian<'a, T: RealField, X, B, const M: usize, Nc>
where T: RealField,
      B: BsplineBasis<T::RealField, X, 1>,
      Nc: Dim,
      DefaultAllocator: Allocator<Const<M>, Nc>
{
    pub geo_map: &'a SplineGeo<'a, T, X, B, M, Nc>,
}

impl <'a, T, const D: usize, const M: usize, Nc> Jacobian<'a, T, [T; D], DeBoorMulti<T, D>, M, Nc>
    where T: RealField + Copy,
          Nc: Dim,
          ShapeConstraint: AreMultipliable<Const<M>, Nc, Dyn, U1>,
          DefaultAllocator: Allocator<Const<M>, Nc>,
          DefaultAllocator: Allocator<Const<M>, Const<D>, Buffer<T> = ArrayStorage<T, M, D>>
{
    /// Evaluates the Jacobian at the parametric point `x`.
    pub fn eval(&self, x: [T; D]) -> SMatrix<T, M, D> {
        let b = &self.geo_map.space.basis;

        // Get nonzero indices and select coefficients
        let (_, idx) = b.eval_deriv_multi_prod(x, 0);
        let c = &self.geo_map.coeffs.select_columns(idx.collect_vec().iter());
        
        // Calculate partial derivatives in each direction and evaluate
        let cols = (0..D).map(|du| {
            let (b_du, _) = b.eval_deriv_multi_prod(x, du);
            c * b_du
        }).collect_vec();
        
        Matrix::from_columns(&cols)
    }
}