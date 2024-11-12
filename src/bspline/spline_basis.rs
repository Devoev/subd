use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;

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
}