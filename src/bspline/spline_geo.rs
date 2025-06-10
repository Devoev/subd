use crate::basis::space::Space;
use crate::basis::traits::{Basis, HgradBasis};
use crate::bspline::global_basis::{BsplineBasis, MultiBsplineBasis};
use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, OMatrix, Point, RealField, SMatrix, U1};

// todo: replace B argument with MultiBsplineBasis directly. or should only a local basis be used?

/// A B-spline geometry embedded [`M`]-dimensional Euclidean space.
/// Each spline geometry is a regular [Spline] where each of the [`M`] components is represented
/// by the same basis [`B`].
/// This is equivalent to using points of size [`M`] for each coefficient 
/// and a single scalar valued basis.
#[derive(Debug, Clone)]
pub struct SplineGeo<'a, T: RealField, X, B, const M: usize> {
    /// Matrix of control points.
    pub control_points: OMatrix<T, Dyn, Const<M>>,

    /// Space of basis functions.
    pub space: &'a Space<T, X, B>,
}

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<'a, T, const M: usize> = SplineGeo<'a, T, T, BsplineBasis<T>, M>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<'a, T, const M: usize> = SplineGeo<'a, T, (T, T), MultiBsplineBasis<T, 2>, M>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<'a, T, const M: usize> = SplineGeo<'a, T, (T, T, T), MultiBsplineBasis<T, 3>, M>;

impl <'a, T: RealField, X, B, const M: usize> SplineGeo<'a, T, X, B, M> {
    /// Constructs a new [`SplineGeo`] from the given `control_points` and `space`.
    pub fn new(control_points: OMatrix<T, Dyn, Const<M>>, space: &'a Space<T, X, B>) -> Self {
        // todo: check if number of control points match space dimension
        SplineGeo { control_points, space }
    }

    /// Constructs a new [`SplineGeo`] from the given matrix `mat` of control points as row vectors.
    pub fn from_matrix<N: Dim>(mat: OMatrix<T, N, Const<M>>, space: &'a Space<T, X, B>) -> Self
        where DefaultAllocator: Allocator<N, Const<M>>
    {
        let c = mat.generic_view((0, 0), (Dyn(mat.nrows()), Const::<M>));
        SplineGeo::new(c.into_owned(), space)
    }
}

// todo: add implementation for LocalBasis as well, or just for local basis

impl <T, X, B, const D: usize, const M: usize> Chart<T, X, D, M> for SplineGeo<'_, T, X, B, M>
    where T: RealField,
          X: Dimensioned<T, D>,
          B: HgradBasis<T, X, D, NumBasis = Dyn>,
{
    fn eval(&self, x: X) -> Point<T, M> {
        let b = self.space.basis.eval(x);
        let c = &self.control_points;
        Point::from((b * c).transpose())
    }

    fn eval_diff(&self, x: X) -> SMatrix<T, M, D> {
        let grads = &self.space.basis.eval_grad(x);
        let c = &self.control_points;
        (grads * c).transpose()
    }
}

impl <'a, T, X, B, const D: usize, const M: usize> Chart<T, X, D, M> for &'a SplineGeo<'a, T, X, B, M>
    where T: RealField,
          X: Dimensioned<T, D>,
          B: HgradBasis<T, X, D, NumBasis = Dyn>,
{
    fn eval(&self, x: X) -> Point<T, M> {
        Point::from((*self).eval(x))
    }

    fn eval_diff(&self, x: X) -> SMatrix<T, M, D> {
        (*self).eval_diff(x)
    }
}

// /// Jacobian matrix of a [`SplineGeo`].
// pub struct Jacobian<'a, T: RealField, X, B, const M: usize>
// where T: RealField,
//       B: Basis<T, X>,
//       DefaultAllocator: Allocator<B::NumComponents, B::NumBasis>
// {
//     pub geo_map: &'a SplineGeo<'a, T, X, B, M>,
// }
//
// impl <'a, T, const D: usize, const M: usize, Nc> Jacobian<'a, T, [T; D], MultiBsplineBasis<T, D>, M>
//     where T: RealField + Copy,
//           Nc: Dim,
//           ShapeConstraint: AreMultipliable<Const<M>, Nc, Dyn, U1>,
//           DefaultAllocator: Allocator<Const<M>, Nc>,
//           DefaultAllocator: Allocator<Const<M>, Const<D>, Buffer<T> = ArrayStorage<T, M, D>>
// {
//     /// Evaluates the Jacobian at the parametric point `x`.
//     pub fn eval(&self, x: [T; D]) -> SMatrix<T, M, D> {
//         let b = &self.geo_map.space.basis;
//
//         // Get nonzero indices and select coefficients
//         let (_, idx) = b.eval_deriv_multi_prod(x, 0);
//         let c = &self.geo_map.coeffs.select_columns(idx.collect_vec().iter());
//
//         // Calculate partial derivatives in each direction and evaluate
//         let cols = (0..D).map(|du| {
//             let (b_du, _) = b.eval_deriv_multi_prod(x, du);
//             c * b_du
//         }).collect_vec();
//
//         Matrix::from_columns(&cols)
//     }
// }