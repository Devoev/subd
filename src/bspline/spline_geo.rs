use crate::space::error::CoeffsSpaceDimError;
use crate::bspline::space::BsplineSpace;
use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, OMatrix, Point, RealField, SMatrix};

/// A [`D`]-variate B-spline geometry embedded [`M`]-dimensional Euclidean space.
/// Each spline geometry is a linear combination where each of the [`M`] components is represented
/// by the same basis [`B`].
/// This is equivalent to using points of size [`M`] for each coefficient 
/// and a single scalar valued basis.
#[derive(Debug, Clone)]
pub struct SplineGeo<'a, T: RealField, const D: usize, const M: usize> {
    /// Matrix of control points.
    pub control_points: OMatrix<T, Dyn, Const<M>>,

    /// Space of scalar B-Splines.
    pub space: &'a BsplineSpace<T, D>
}

/// A spline curve in [`M`] dimensions.
pub type SplineCurve<'a, T, const M: usize> = SplineGeo<'a, T, 1, M>;

/// A spline surface in [`M`] dimensions.
pub type SplineSurf<'a, T, const M: usize> = SplineGeo<'a, T, 2, M>;

/// A spline volume in [`M`] dimensions.
pub type SplineVol<'a, T, const M: usize> = SplineGeo<'a, T, 3, M>;

/// A [`D`]-variate spline geometry in [`M`]-dimensions.
pub type MultiSplineGeo<'a, T, const D: usize, const M: usize> = SplineGeo<'a, T, D, M>;

impl <'a, T: RealField, const D: usize, const M: usize> SplineGeo<'a, T, D, M> {
    /// Constructs a new [`SplineGeo`] from the given `control_points` and `space`.
    ///
    /// # Errors
    /// Will return an error if the number of rows of `control_points`
    /// does not match the dimension of `space`.
    pub fn new(control_points: OMatrix<T, Dyn, Const<M>>, space: &'a BsplineSpace<T, D>) -> Result<Self, CoeffsSpaceDimError> {
        match control_points.nrows() == space.dim() {
            true => Ok(SplineGeo { control_points, space }),
            false => Err(CoeffsSpaceDimError { num_coeffs: control_points.nrows(), dim_space: space.dim() })
        }
    }

    /// Constructs a new [`SplineGeo`] from the given matrix `mat` of control points as row vectors.
    pub fn from_matrix<N: Dim>(mat: OMatrix<T, N, Const<M>>, space: &'a BsplineSpace<T, D>) -> Result<Self, CoeffsSpaceDimError>
        where DefaultAllocator: Allocator<N, Const<M>>
    {
        let c = mat.generic_view((0, 0), (Dyn(mat.nrows()), Const::<M>));
        SplineGeo::new(c.into_owned(), space)
    }
}

impl <T, const D: usize, const M: usize> Chart<T> for SplineGeo<'_, T, D, M>
    where T: RealField + Copy
{
    type Coord = [T; D];
    type ParametricDim = Const<D>;
    type GeometryDim = Const<M>;

    fn eval(&self, x: [T; D]) -> Point<T, M> {
        let (b, idx) = self.space.eval_local_with_idx(x);
        // todo: replace collect_vec().iter() somehow
        let c = &self.control_points.select_rows(idx.collect_vec().iter());
        Point::from((b * c).transpose())
    }

    fn eval_diff(&self, x: [T; D]) -> SMatrix<T, M, D> {
        let (grad_b, idx) = self.space.eval_grad_local_with_idx(x);
        // todo: replace collect_vec().iter() somehow
        let c = &self.control_points.select_rows(idx.collect_vec().iter());
        (grad_b * c).transpose()
    }
}

impl <'a, T, const D: usize, const M: usize> Chart<T> for &'a SplineGeo<'a, T, D, M>
    where T: RealField + Copy
{
    type Coord = [T; D];
    type ParametricDim = Const<D>;
    type GeometryDim = Const<M>;

    fn eval(&self, x: [T; D]) -> Point<T, M> {
        Point::from((*self).eval(x))
    }

    fn eval_diff(&self, x: [T; D]) -> SMatrix<T, M, D> {
        (*self).eval_diff(x)
    }
}