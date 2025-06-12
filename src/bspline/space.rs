use std::iter::zip;
use itertools::Itertools;
use nalgebra::RealField;
use crate::basis::space::Space;
use crate::bspline::global_basis::{BsplineBasis, MultiBsplineBasis, MultiBsplineBasis2d};
use crate::knots::knot_vec::KnotVec;
// todo: add more type aliases and proper documentation (type parameters, special cases...)

/// Function space of [`D`]-variate scalar B-Spline basis functions.
pub type BsplineSpace<T, X, const D: usize> = Space<T, X, MultiBsplineBasis<T, D>, D>;

/// Function space of [`D`]-variate vector B-Spline basis functions in 2D.
pub type BsplineSpace2d<T, X, const D: usize> = Space<T, X, MultiBsplineBasis2d<T, D>, D>;

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
            BsplineBasis::new(knots, n, p)
        }).collect_array().unwrap();
        BsplineSpace::new(MultiBsplineBasis::new(bases))
    }
}