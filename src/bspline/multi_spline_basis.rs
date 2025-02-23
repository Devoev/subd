use crate::bspline::spline_basis::SplineBasis;
use crate::knots::index::{MultiIndex, Strides};
use crate::knots::knot_span::{KnotSpan, MultiKnotSpan};
use crate::bspline::basis::Basis;
use itertools::Itertools;
use nalgebra::{DVector, RealField};
use std::iter::zip;

/// A [`D`]-variate knot vector.
#[derive(Debug, Clone)]
pub struct MultiSplineBasis<T: RealField, const D: usize>(pub(crate) [SplineBasis<T>; D]);

impl<T: RealField + Copy, const D: usize> MultiSplineBasis<T, D> {

    /// Constructs a new [`MultiSplineBasis`] from the given univariate spaces.
    pub fn new(spaces: [SplineBasis<T>; D]) -> Self {
        MultiSplineBasis(spaces)
    }
    
    /// Constructs an open [`MultiSplineBasis`] with `n[i] + p[i] + 1` for each direction.
    pub fn open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        let arr: [SplineBasis<T>; D] = zip(n, p)
            .map(|(n, p)| SplineBasis::open_uniform(n, p))
            .collect_vec()
            .try_into()
            .unwrap();
        arr.into()
    }
}

impl <'a, T: RealField + Copy, const D: usize> Basis<'a, [T; D], T, MultiIndex<usize, D>> for MultiSplineBasis<T, D> {
    fn num(&self) -> usize {
        self.n().iter().product()
    }

    fn find_span(&'a self, t: [T; D]) -> Result<KnotSpan<'a, MultiIndex<usize, D>, Self>, ()> {
        MultiKnotSpan::find(self, t)
    }

    fn eval(&self, t: [T; D]) -> DVector<T> {
        zip(&self.0, t)
            .map(|(knot_vec, ti)| knot_vec.eval(ti))
            .reduce(|acc, bi| acc.kronecker(&bi))
            .expect("Dimension D must be greater than 0!")
    }
}

impl<T: RealField + Copy, const D : usize> MultiSplineBasis<T, D> {
    
    /// Return the number of basis functions per parametric direction.
    pub fn n(&self) -> [usize; D] {
        self.0.iter().map(|knot_vec| knot_vec.n).collect_array().unwrap()
    }
    
    /// Return the degrees of basis functions per parametric direction.
    pub fn p(&self) -> [usize; D] {
        self.0.iter().map(|knot_vec| knot_vec.p).collect_array().unwrap()
    }
    
    /// Returns the strides for the multi index ordering of the basis functions.
    pub fn strides(&self) -> Strides<usize, D> {
        Strides::from_dims(self.n())
    }
}

impl<T: RealField + Copy, const D: usize> From<[SplineBasis<T>; D]> for MultiSplineBasis<T, D> {
    fn from(value: [SplineBasis<T>; D]) -> Self {
        MultiSplineBasis::new(value)
    }
}