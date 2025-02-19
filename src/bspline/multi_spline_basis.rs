use crate::bspline::spline_basis::SplineBasis;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector, RealField};
use std::iter::{zip, Map};
use std::slice::Iter;

/// A [D]-variate spline basis.
#[derive(Clone, Debug)]
pub struct MultiSplineBasis<T : RealField, const D : usize> {
    pub univariate_bases: [SplineBasis<T>; D]
}

impl<T : RealField + Copy, const D : usize> MultiSplineBasis<T, D> {

    /// Constructs a new [`MultiSplineBasis`].
    pub fn new(univariate_bases: [SplineBasis<T>; D]) -> Self {
        MultiSplineBasis { univariate_bases }
    }

    /// Constructs a new [`MultiSplineBasis`] on an open knot vector of size `n+p+1`.
    pub fn open(n: [usize; D], p: [usize; D]) -> Self {
        let bases: [SplineBasis<T>; D] = zip(n, p)
            .map(|(n, p)| SplineBasis::open(n, p))
            .collect_vec()
            .try_into()
            .unwrap();
        Self::new(bases)
    }
    
    /// Returns the total number of basis functions.
    pub fn num(&self) -> usize {
        self.univariate_bases.iter().map(|basis| basis.n).product()
    }
    
    /// Returns an iterator over the number of basis functions in each parametric direction.
    pub fn n(&self) -> impl Iterator<Item=usize> + '_ {
        self.univariate_bases.iter().map(|basis| basis.n)
    }
    
    /// Returns an iterator over the degrees of basis functions in each parametric direction.
    pub fn p(&self) -> impl Iterator<Item=usize> + '_ {
        self.univariate_bases.iter().map(|basis| basis.p)
    }
}

impl <T : RealField + Copy> MultiSplineBasis<T, 1> {

    /// Evaluates the non-vanishing basis functions at the parametric point `t`.
    pub fn eval_curve(&self, t: T) -> DVector<T> {
        self.univariate_bases[0].eval(t)
    }
}

impl <T : RealField + Copy, const D : usize> MultiSplineBasis<T, D> {
    
    /// Evaluates the non-vanishing basis functions at the parametric point `t`.
    pub fn eval(&self, t: [T; D]) -> DVector<T> {
        zip(&self.univariate_bases, t)
            .map(|(basis, ti)| basis.eval(ti))
            .reduce(|acc, bi| acc.kronecker(&bi))
            .expect("Dimension D must be greater than 0!")
    }
}