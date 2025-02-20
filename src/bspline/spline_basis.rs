use crate::knots::knot_vec::KnotVec;
use nalgebra::{DVector, RealField};

/// A B-spline basis of `n` basis functions of degree `p`.
#[derive(Debug, Clone)]
pub struct SplineBasis<T : RealField> {
    
    /// Knot vector for the allocation of the basis functions.
    pub knots: KnotVec<T>,
    
    /// Number of basis functions.
    pub n: usize,
    
    /// Degree of basis functions.
    pub p: usize,
}

impl<T : RealField + Copy> SplineBasis<T> {

    /// Constructs a new [`SplineBasis`].
    pub fn new(knots: KnotVec<T>, n: usize, p: usize) -> Self {
        SplineBasis { knots, n, p }
    }
    
    /// Constructs a new [`SplineBasis`] on an open knot vector of size `n+p+1`.
    pub fn open(n: usize, p: usize) -> Self {
        Self::new(KnotVec::open_uniform(n, p), n, p)
    }
}

impl<T : RealField + Copy> SplineBasis<T> {
    
    /// Evaluates the `p+1` non-vanishing basis functions at the parametric point `t`.
    pub fn eval(&self, t: T) -> DVector<T> {
        let span = self.knots.find_span(t)
            .expect("Parametric value is outside of knot vector.");
        let mut left = vec![T::zero(); self.p + 1];
        let mut right = vec![T::zero(); self.p + 1];
        let mut b = DVector::zeros(self.p + 1);
        b[0] = T::one();

        for i in 1..=self.p {
            left[i] = t - self.knots[span.index - i + 1];
            right[i] = self.knots[span.index + i] - t;
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