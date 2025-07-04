use std::iter::zip;
use itertools::Itertools;
use nalgebra::RealField;
use crate::basis::space::Space;
use crate::bspline::de_boor::{DeBoor, MultiDeBoor, DeBoorVec2d};
use crate::knots::knot_vec::KnotVec;

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
}

impl <T: RealField + Copy, X, const D: usize> BsplineSpace<T, X, D> {
    /// Constructs an *open* and *uniform* [`BsplineSpace`]
    /// with `n[i]+p[i]+1` knots for each parametric direction.
    ///
    /// # Examples
    /// ```
    /// # use subd::bspline::space::BsplineSpace;
    ///
    /// let n = [5, 4, 6];
    /// let p = [1, 2, 2];
    /// let space = BsplineSpace::new_open_uniform(n, p);
    /// ```
    pub fn new_open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        let bases = zip(n, p).map(|(n, p)| {
            let knots = KnotVec::new_open_uniform(n, p);
            DeBoor::new(knots, n, p)
        }).collect_array().unwrap();
        BsplineSpace::new(MultiDeBoor::new(bases))
    }
}