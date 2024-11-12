use nalgebra::RealField;
use log::debug;
use crate::knots::knot_vec::KnotVec;

pub struct SplineBasis<T : RealField> {
    pub knots: KnotVec<T>,
    pub n: usize,
    pub p: usize,
}

impl<T : RealField> SplineBasis<T> {

    pub fn new(knots: KnotVec<T>, n: usize, p: usize) -> Self {
        SplineBasis { knots, n, p }
    }
}

impl<T : RealField> SplineBasis<T> {

    pub fn find_span(&self, t: T) -> usize {
        if t == self.knots[self.n + 1] {
            return self.n-1;
        }

        let idx = self.knots.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        match idx {
            Ok(i) => { i }
            Err(i) => { i - 1 }
        }
    }
}