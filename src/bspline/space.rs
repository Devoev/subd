use crate::basis::space::Space;
use crate::bspline::de_boor::{DeBoor, DeBoorVec2d, MultiDeBoor};
use crate::index::dimensioned::{DimShape, MultiRange, Strides};
use crate::index::multi_index::MultiIndex;
use crate::knots::knot_vec::KnotVec;
use itertools::Itertools;
use nalgebra::RealField;
use std::collections::HashSet;
use std::iter::zip;

// todo: the vector valued spaces are not special cases. Change this somehow?

/// Function space of [`D`]-variate B-Splines.
/// The basis evaluations are computed with variations of [`DeBoor`].
///
/// # Special cases
/// - [`BsplineSpace1d`], [`BsplineSpace2d`], [`BsplineSpace3d`] for scalar valued
///   basis functions. Univariate, bivariate and trivariate respectively.
/// - [`BsplineSpaceVec2d`] for vector valued basis functions in 2D.
///
/// # Type parameters
/// - [`T`]: Scalar type for coefficients.
/// - [`X`]: Type of parametric values in the reference domain.
/// - [`D`]: Dimension of the parametric domain.
pub type BsplineSpace<T, X, const D: usize> = Space<T, X, MultiDeBoor<T, D>, D>;

/// Functions space of univariate scalar B-Splines.
pub type BsplineSpace1d<T> = BsplineSpace<T, T, 1>;

/// Function space of bivariate scalar B-Splines.
pub type BsplineSpace2d<T> = BsplineSpace<T, (T, T), 2>;

/// Function space of trivariate scalar B-Splines.
pub type BsplineSpace3d<T> = BsplineSpace<T, (T, T, T), 3>;

/// Function space of 2D vector B-Splines.
pub type BsplineSpaceVec2d<T> = Space<T, (T, T), DeBoorVec2d<T>, 2>;

impl<T, X, const D: usize> BsplineSpace<T, X, D> {
    /// Returns an array of the knot vectors for each parametric direction.
    pub fn knots(&self) -> [&KnotVec<T>; D] {
        self.basis.bases.iter().map(|b| &b.knots).collect_array().unwrap()
    }

    /// Returns the [`DimShape`] for the basis function indexing.
    pub fn dim_shape(&self) -> DimShape<D> {
        let dims = self.basis.bases.iter()
            .map(|b| b.num_basis)
            .collect_array()
            .unwrap();
        DimShape(dims)
    }

    /// Returns an iterator over all multi-indices of basis functions.
    pub fn basis_indices(&self) -> MultiRange<[usize; D]> {
        self.dim_shape().multi_range()
    }

    /// Returns a set over all linear indices corresponding to functions at the boundary.
    pub fn boundary_indices(&self) -> HashSet<usize> {
        let shape = self.dim_shape();
        let strides = Strides::from(shape);
        shape.boundary_indices()
            .iter()
            .map(|idx| idx.into_lin(&strides))
            .collect()
    }
}

impl <T: RealField + Copy, X, const D: usize> BsplineSpace<T, X, D> {
    /// Constructs an *open* and *uniform* [`BsplineSpace`]
    /// with `n[i]+p[i]+1` knots for each parametric direction.
    ///
    /// # Examples
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use subd::bspline::space::BsplineSpace;
    /// # use subd::knots::knot_vec::KnotVec;
    ///
    /// let n = [5, 4, 6];
    /// let p = [1, 2, 2];
    /// let space = BsplineSpace::<f64, [f64; 3], 3>::new_open_uniform(n, p);
    ///
    /// let [KnotVec(xi_1), KnotVec(xi_2), KnotVec(xi_3)] = &space.knots();
    /// assert_abs_diff_eq!(xi_1.as_slice(), [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0].as_slice());
    /// assert_abs_diff_eq!(xi_2.as_slice(), [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0].as_slice());
    /// assert_abs_diff_eq!(xi_3.as_slice(), [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0].as_slice());
    /// ```
    pub fn new_open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        let bases = zip(n, p).map(|(n, p)| {
            let knots = KnotVec::new_open_uniform(n, p);
            DeBoor::new(knots, n, p)
        }).collect_array().unwrap();
        BsplineSpace::new(MultiDeBoor::new(bases))
    }
}