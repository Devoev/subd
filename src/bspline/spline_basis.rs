use crate::bspline::basis::BsplineBasis;
use crate::knots::knot_span::{KnotSpan, KnotSpan1};
use crate::knots::knot_vec::KnotVec;
use itertools::chain;
use nalgebra::{DVector, RealField};
use std::ops::RangeInclusive;
use std::vec;

/// The space of `n` B-spline basis functions of degree `p`, allocated on the `knots`.
#[derive(Debug, Clone)]
pub struct SplineBasis<T: RealField> {
    /// The vector of knot values.
    pub(crate) knots: KnotVec<T>,
    
    /// Number of basis functions.
    pub n: usize,
    
    /// Degree of basis functions.
    pub p: usize
}

impl<T : RealField + Copy> SplineBasis<T> {

    /// Constructs a new [`SplineBasis`].
    /// 
    /// If the values of `n` and `p` don't match the length of the knots,
    /// [`None`] is returned.
    pub fn new(knots: KnotVec<T>, n: usize, p: usize) -> Option<Self> {
        (knots.len() == n + p + 1).then_some(SplineBasis { knots, n, p })
    }
    
    /// Constructs an open [`SplineBasis`] with given `internal` knot values.
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

    /// Constructs an open uniform [`SplineBasis`].
    pub fn open_uniform(n: usize, p: usize) -> Self {
        let internal = KnotVec::uniform(n-p+1);
        let knots = chain!(
            std::iter::repeat_n(T::zero(), p),
            internal.0,
            std::iter::repeat_n(T::one(), p)
        ).collect();
        
        SplineBasis { knots: KnotVec(knots), n, p }
    }
}

impl<T: RealField + Copy> SplineBasis<T> {
    pub(crate) fn find_span(&self, t: T) -> Result<KnotSpan<usize>, ()> {
        KnotSpan1::find(self, t)
    }

    fn nonzero(&self, span: &KnotSpan<usize>) -> RangeInclusive<usize> {
        span.nonzero_indices(self.p)
    }

    pub(crate) fn eval(&self, t: T, span: &KnotSpan1) -> DVector<T> {
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

impl <T: RealField + Copy> BsplineBasis<T, T> for SplineBasis<T> {
    type NonzeroIndices = RangeInclusive<usize>;

    fn len(&self) -> usize {
        self.n
    }

    fn eval_nonzero(&self, x: T) -> (DVector<T>, Self::NonzeroIndices) {
        let span = self.find_span(x).unwrap();
        (self.eval(x, &span), self.nonzero(&span))
    }
}