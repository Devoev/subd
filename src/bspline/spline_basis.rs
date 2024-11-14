use crate::knots::knot_vec::KnotVec;
use nalgebra::{DVector, RealField, Vector};

/// A B-spline basis of `n` basis functions of degree `p`.
pub struct SplineBasis<T : RealField> {
    pub knots: KnotVec<T>,
    pub n: usize,
    pub p: usize,
}

impl<T : RealField> SplineBasis<T> {

    /// Constructs a new `SplineBasis`.
    pub fn new(knots: KnotVec<T>, n: usize, p: usize) -> Self {
        SplineBasis { knots, n, p }
    }
}

impl<T : RealField + Copy> SplineBasis<T> {
    
    /// Finds the index `i` such that `knots[i] <= t < knots[i+1]`.
    pub fn find_span(&self, t: T) -> Result<usize, ()> {
        if t < self.knots.first() || t > self.knots.last() { return Err(()) }
        
        if t == self.knots[self.n + 1] {
            return Ok(self.n - 1);
        }

        let idx = self.knots.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        match idx {
            Ok(i) => { Ok(i) }
            Err(i) => { Ok(i - 1) }
        }
    }

    pub fn eval(&self, t: T) -> Vec<T> {
        let idx = self.find_span(t).unwrap();
        let mut left = vec![T::zero(); self.p + 1];
        let mut right = vec![T::zero(); self.p + 1];
        let mut B = vec![T::zero(); self.p + 1];
        B[0] = T::one();
        
        for i in 0..self.p {
            left[i + 1] = t - self.knots[idx - i];
            right[i + 1] = self.knots[idx + i + 1] - t;
            let mut saved = T::zero();

            for j in 0..=i {
                let tmp = B[j] / (right[j+1] + left[i-j+1]);
                B[j] = saved + right[j+1]*tmp;
                saved = left[i-j+1]*tmp;
            }
            B[i+1] = saved;
        }
        B
    }
}