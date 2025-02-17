use std::ops::RangeInclusive;
use nalgebra::RealField;
use crate::knots::knot_vec::KnotVec;

/// A knot span `[xi[i], xi[i+1])`.
#[derive(Debug, Clone)]
pub struct KnotSpan<'a, T: RealField> {

    /// The knot vector.
    knots: &'a KnotVec<T>,

    /// The span index.
    pub index: usize
}

// todo: make n and p part of KnotVec
impl<'a, T: RealField + Copy> KnotSpan<'a, T> {

    /// Constructs a new [KnotSpan].
    pub fn new(knots: &'a KnotVec<T>, index: usize) -> Self {
        KnotSpan { knots, index }
    }

    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self, p: usize) -> RangeInclusive<usize> {
        self.index - p..=self.index
    }
}

impl <T: RealField + Copy> KnotVec<T> {

    /// Finds the [KnotSpan] containing the parametric value `t`.
    ///
    /// # Arguments
    /// * `t` - Parametric value.
    /// * `n` - Number of basis functions.
    ///
    pub fn find_span(&self, t: T, n: usize) -> Result<KnotSpan<T>, ()> {
        if t < self.first() || t > self.last() { return Err(()) }

        if t == self[n + 1] {
            return Ok(KnotSpan::new(self, n - 1));
        }

        let idx = self.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        let idx = match idx {
            Ok(i) => { i }
            Err(i) => { i - 1 }
        };
        Ok(KnotSpan::new(self, idx))
    }
}