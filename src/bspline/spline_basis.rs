use crate::knots::knot_vec::KnotVec;
use crate::knots::multi_knot_vec::MultiKnotVec;
use nalgebra::{DVector, RealField};
use crate::knots::knots_trait::Knots;

/// A set of B-spline basis functions.
#[derive(Debug, Clone)]
pub struct SplineBasis<K> {
    /// Knot vector for the allocation of the basis functions.
    pub knots: K
}

impl<K> SplineBasis<K> {
    /// Constructs a new [`SplineBasis`].
    pub fn new(knots: K) -> Self { SplineBasis { knots } }
}

/// A multivariate spline basis.
pub type MultiSplineBasis<T, const D: usize> = SplineBasis<MultiKnotVec<T, D>>;

/// A univariate spline basis.
pub type SplineBasis1<T> = SplineBasis<KnotVec<T>>;

/// A bivariate spline basis.
pub type SplineBasis2<T> = SplineBasis<MultiKnotVec<T, 2>>;

impl <T: RealField + Copy> SplineBasis1<T> {
    /// Constructs a new [`SplineBasis`] on an open uniform [`KnotVec`] of size `n+p+1`.
    pub fn open_uniform(n: usize, p: usize) -> Self {
        SplineBasis::new(KnotVec::open_uniform(n, p))
    }
    
    /// Returns the total number of basis functions.
    pub fn num(&self) -> usize {
        self.knots.n
    }

    /// Evaluates this basis at `t`.
    pub fn eval(&self, t: T) -> DVector<T> {
        self.knots.eval_basis(t)
    }
}

impl<T: RealField + Copy, const D: usize> MultiSplineBasis<T, D> {
    /// Constructs a new [`MultiSplineBasis`] from given univariate bases.
    pub fn from_bases(bases: [SplineBasis1<T>; D]) -> Self {
        MultiSplineBasis::new(MultiKnotVec(bases.map(|b| b.knots)))
    }

    /// Constructs a new [`MultiSplineBasis`] on an open uniform knot vector of size `n+p+1`.
    pub fn open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        MultiSplineBasis::new(MultiKnotVec::open_uniform(n, p))
    }

    /// Returns the total number of basis functions.
    pub fn num(&self) -> usize {
        self.knots.0.iter().map(|basis| basis.n).product()
    }

    /// Returns an iterator over the number of basis functions in each parametric direction.
    pub fn n(&self) -> impl Iterator<Item=usize> + '_ {
        self.knots.0.iter().map(|basis| basis.n)
    }

    /// Returns an iterator over the degrees of basis functions in each parametric direction.
    pub fn p(&self) -> impl Iterator<Item=usize> + '_ {
        self.knots.0.iter().map(|basis| basis.p)
    }

    /// Evaluates this basis at `t`.
    pub fn eval(&self, t: [T; D]) -> DVector<T> {
        self.knots.eval_basis(t)
    }
}