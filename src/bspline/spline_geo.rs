use crate::bspline::de_boor::{DeBoor, DeBoorBi, DeBoorMulti};
use crate::bspline::spline::Spline;
use nalgebra::{ArrayStorage, Const, DefaultAllocator, Dim, DimName, Dyn, Matrix, Point, RealField, SMatrix, SVector, U1, U2};
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{AreMultipliable, DimEq, ShapeConstraint};
use crate::bspline::basis::BsplineBasis;
use crate::cells::chart::Chart;
use crate::knots::knot_span::KnotSpan;
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

/// Jacobian matrix of a [`SplineGeo`].
pub struct Jacobian<'a, T: RealField, X, B, const M: usize, Nc>
where T: RealField,
      B: BsplineBasis<T::RealField, X, 1>,
      Nc: Dim,
      DefaultAllocator: Allocator<Const<M>, Nc>
{
    pub geo_map: &'a SplineGeo<'a, T, X, B, M, Nc>,
}

impl <'a, T, const M: usize, Nc> Jacobian<'a, T, [T; 2], DeBoorBi<T>, M, Nc>
    where T: RealField + Copy,
          Nc: Dim,
          ShapeConstraint: AreMultipliable<Const<M>, Nc, Dyn, U1>,
          DefaultAllocator: Allocator<Const<M>, Nc>,
          DefaultAllocator: Allocator<Const<M>, U2, Buffer<T> = ArrayStorage<T, M, 2>>
{
    // todo: implement for arbitrary number of parameters and rework algorithm
    pub fn eval(&self, x: [T; 2]) -> SMatrix<T, M, 2> {
        let b = &self.geo_map.space.basis;
        let b1 = &b.b1;
        let b2 = &b.b2;
        let span1 = KnotSpan::find(&b1.knots, b1.n, x[0]).unwrap();
        let span2 = KnotSpan::find(&b2.knots, b2.n, x[1]).unwrap();
        
        // Evaluate bases and derivatives in both parametric directions
        let b1_eval = b1.eval_derivs_with_span::<1>(x[0], span1);
        let b2_eval = b2.eval_derivs_with_span::<1>(x[1], span2);

        // Split up basis and derivative evaluations
        let b1 = b1_eval.row(0);
        let b1_du = b1_eval.row(1);
        let b2 = b2_eval.row(0);
        let b2_dv = b2_eval.row(1);

        // Calculate partial derivatives of bivariate basis
        let b_du = b2.kronecker(&b1_du).transpose();
        let b_dv = b2_dv.kronecker(&b1).transpose();
        
        // todo: this should generally fail, because only the nonzero coefficients should be taken
        let c = &self.geo_map.coeffs;
        let cols = &[c.clone() * b_du, c.clone() * b_dv];
        Matrix::from_columns(cols)
    }
}