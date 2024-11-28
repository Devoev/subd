use crate::bspline::spline_basis::SplineBasis;
use itertools::Itertools;
use nalgebra::{DMatrix, RealField};
use std::iter::zip;

/// A multivariate spline basis.
#[derive(Clone, Debug)]
pub struct MultivariateSplineBasis<T : RealField, const D : usize> {
    univariate_bases: [SplineBasis<T>; D]
}

impl<T : RealField + Copy, const D : usize> MultivariateSplineBasis<T, D> {

    /// Constructs a new [`MultivariateSplineBasis`].
    pub fn new(univariate_bases: [SplineBasis<T>; D]) -> Self {
        MultivariateSplineBasis { univariate_bases }
    }

    /// Constructs a new [`MultivariateSplineBasis`] on an open knot vector of size `n+p+1`.
    pub fn open(n: [usize; D], p: [usize; D]) -> Self {
        let bases: [SplineBasis<T>; D] = zip(n, p)
            .map(|(n, p)| SplineBasis::open(n, p))
            .collect_vec()
            .try_into()
            .unwrap();
        Self::new(bases)
    }
}

impl<T : RealField + Copy> MultivariateSplineBasis<T, 2> {
    
    /// Evaluates the non-vanishing basis functions at the parametric point `t`.
    pub fn eval(&self, t: [T; 2]) -> DMatrix<T> {
        let mut B = zip(&self.univariate_bases, t)
            .map(|(basis, ti)| basis.eval(ti));
        
        let Bx = B.next().unwrap();
        let By = B.next().unwrap();
        
        Bx * By.transpose()
    }
}