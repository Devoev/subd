use std::iter::zip;
use crate::bspline::basis::BsplineBasis;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use itertools::{chain, Itertools};
use nalgebra::{DVector, RealField};
use std::ops::RangeInclusive;
use std::vec;
use crate::bspline::tensor_prod::MultiProd;

/// De-Boor algorithm for the computation of the B-Spline basis
/// of dimension `n` and degree `p`.
#[derive(Debug, Clone)]
pub struct DeBoor<T: RealField> {
    /// The vector of knot values.
    pub(crate) knots: KnotVec<T>,
    
    /// Number of basis functions.
    pub n: usize,
    
    /// Degree of basis functions.
    pub p: usize
}

impl<T : RealField + Copy> DeBoor<T> {

    /// Constructs a new [`DeBoor`].
    /// 
    /// If the values of `n` and `p` don't match the length of the knots,
    /// [`None`] is returned.
    pub fn new(knots: KnotVec<T>, n: usize, p: usize) -> Option<Self> {
        (knots.len() == n + p + 1).then_some(DeBoor { knots, n, p })
    }
    
    /// Constructs an open [`DeBoor`] with given `internal` knot values.
    /// 
    /// If internal knots are not inside `(0,1)`, [`None`] is returned.
    pub fn open(internal: KnotVec<T>, n: usize, p: usize) -> Option<Self> {
        let knots = chain!(
            std::iter::repeat_n(T::zero(), p+1),
            internal.0,
            std::iter::repeat_n(T::one(), p+1)
        ).collect();
        
        Self::new(KnotVec(knots), n, p)
    }

    /// Constructs an open uniform [`DeBoor`].
    pub fn open_uniform(n: usize, p: usize) -> Self {
        let internal = KnotVec::uniform(n-p+1);
        let knots = chain!(
            std::iter::repeat_n(T::zero(), p),
            internal.0,
            std::iter::repeat_n(T::one(), p)
        ).collect();
        
        DeBoor { knots: KnotVec(knots), n, p }
    }

    /// Evaluates the basis functions inside the given `span` at the parametric point `t`.
    pub(crate) fn eval_with_span(&self, t: T, span: KnotSpan) -> DVector<T> {
        let knots = &self.knots;
        let mut left = vec![T::zero(); self.p + 1];
        let mut right = vec![T::zero(); self.p + 1];
        let mut b = DVector::zeros(self.p + 1);
        b[0] = T::one();

        for i in 1..=self.p {
            left[i] = t - knots[span.0 - i + 1];
            right[i] = knots[span.0 + i] - t;
            let mut saved = T::zero();

            for j in 0..i {
                let tmp = b[j] / (right[j+1] + left[i-j]);
                b[j] = saved + right[j+1]*tmp;
                saved = left[i-j]*tmp;
            }
            b[i] = saved;
        }
        b
    }
}

impl <T: RealField + Copy> BsplineBasis<T, T> for DeBoor<T> {
    type NonzeroIndices = RangeInclusive<usize>;

    fn len(&self) -> usize {
        self.n
    }

    fn eval_nonzero(&self, x: T) -> (DVector<T>, Self::NonzeroIndices) {
        let span = KnotSpan::find(&self.knots, self.n, x).unwrap();
        (self.eval_with_span(x, span), span.nonzero_indices(self.p))
    }
}

/// Tensor product generalization of [de Boors algorithm](DeBoor) for the
/// computation of a [`D`]-variate B-Spline basis.
pub type MultiDeBoor<T, const D: usize> = MultiProd<T, DeBoor<T>, D>;

impl<T: RealField + Copy, const D: usize> MultiDeBoor<T, D> {
    /// Constructs an open [`MultiDeBoor`] with `n[i] + p[i] + 1` for each direction.
    pub fn open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        let arr: [DeBoor<T>; D] = zip(n, p)
            .map(|(n, p)| DeBoor::open_uniform(n, p))
            .collect_vec()
            .try_into()
            .unwrap();

        MultiDeBoor::new(arr)
    }
}

impl<T: RealField + Copy, const D : usize> MultiDeBoor<T, D> {
    /// Return the degrees of basis functions per parametric direction.
    pub fn degrees(&self) -> [usize; D] {
        self.bases.iter().map(|b| b.p).collect_array().unwrap()
    }
}
