use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;
use std::ops::RangeInclusive;

/// A knot span represented by a knot index `i`, 
/// such that `xi[i] <= i < xi[i+1]`, where
/// `xi` is the knot vector.
/// 
/// The span is thus the half-open interval `[xi[i], xi[i+1])`.
#[derive(Debug, Copy, Clone)]
pub struct KnotSpan(pub(crate) usize);

impl KnotSpan {
    /// Finds the [`KnotSpan`] containing the given parametric value `t`.
    ///
    /// # Errors
    /// Will return an error if the parametric value lies outside the given `knots`.
    ///
    /// # Examples
    /// ```
    /// use subd::knots::knot_span::KnotSpan;
    /// use subd::knots::knot_vec::KnotVec;
    ///
    /// let xi = KnotVec::uniform(10);
    /// let span = KnotSpan::find(&xi, 4, 0.5);
    /// ```
    pub fn find<T: RealField + Copy>(knots: &KnotVec<T>, n: usize, t: T) -> Result<Self, ()> {
        if !knots.range().contains(&t) { return Err(()) }

        // todo: update the usage of n in this algorithm and DEBUG in general
        if t == knots[n + 1] {
            return Ok(KnotSpan(n - 1));
        }

        let idx = knots.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        let idx = match idx {
            Ok(i) => { i }
            Err(i) => { i - 1 }
        };
        Ok(KnotSpan(idx))
    }
    
    /// Returns a range over all indices of basis functions of degree `p`
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self, p: usize) -> RangeInclusive<usize> {
        self.0 - p..=self.0
    }
}