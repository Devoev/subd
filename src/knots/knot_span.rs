use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;
use std::ops::RangeInclusive;
use crate::knots::error::OutsideKnotRangeError;

/// The half-open (possibly empty) interval `[xi[i], xi[i+1])`, where
/// `xi` is the knot vector.
/// 
/// It is represented by a single index `i`, 
/// such that `xi[i] <= i < xi[i+1]`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct KnotSpan(pub(crate) usize);

impl KnotSpan {
    // todo: only works for open knot vectors, hence the dependence on `n`. change this?
    /// Finds the **non-empty** [`KnotSpan`] containing the given parametric value `t`.
    ///
    /// # Errors
    /// Will return an error if the parametric value lies outside the given `knots`.
    ///
    /// # Examples
    /// ```
    /// # use subd::knots::knot_span::KnotSpan;
    /// # use subd::knots::knot_vec::KnotVec;
    ///
    /// let xi = KnotVec::new_uniform(10);
    /// let span = KnotSpan::find(&xi, 4, 0.5);
    /// ```
    pub fn find<T: RealField + Copy>(knots: &KnotVec<T>, n: usize, t: T) -> Result<Self, OutsideKnotRangeError> {
        if !knots.range().contains(&t) { return Err(OutsideKnotRangeError) }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_span() {
        //           indices:     0    1     2    3     4    5    6
        let knots = KnotVec(vec![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        let n = 5; // n = 5, p = 1

        // Find span [0.5, 0.75)
        let idx = KnotSpan::find(&knots, n, 0.5);
        assert_eq!(idx, Ok(KnotSpan(3)));

        let idx = KnotSpan::find(&knots, n, 0.67);
        assert_eq!(idx, Ok(KnotSpan(3)));

        // Find span [0.75, 1.0)
        let idx = KnotSpan::find(&knots, n, 0.75);
        assert_eq!(idx, Ok(KnotSpan(4)));

        // Find at boundaries
        let idx = KnotSpan::find(&knots, n, 0.0);
        assert_eq!(idx, Ok(KnotSpan(1)));

        let idx = KnotSpan::find(&knots, n, 1.0);
        assert_eq!(idx, Ok(KnotSpan(4)));

        // Outside boundaries
        let idx = KnotSpan::find(&knots, n, -0.1);
        assert_eq!(idx, Err(OutsideKnotRangeError));

        let idx = KnotSpan::find(&knots, n, 1.1);
        assert_eq!(idx, Err(OutsideKnotRangeError));
    }
}