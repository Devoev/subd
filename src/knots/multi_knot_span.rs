use std::ops::RangeInclusive;
use itertools::Itertools;
use nalgebra::RealField;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use crate::knots::multi_knot_vec::MultiKnotVec;

/// A `D`-variate knot span.
#[derive(Clone, Debug)]
pub struct MultiKnotSpan<'a, T: RealField, const D: usize> {
    /// The multivariate knot vector.
    knots: &'a MultiKnotVec<T, D>,
    
    /// The span indices for each knot vector.
    pub indices: [usize; D]
}

impl<'a, T: RealField + Copy, const D: usize> MultiKnotSpan<'a, T, D> {

    /// Constructs a new [MultiKnotSpan].
    pub fn new(knots: &'a MultiKnotVec<T, D>, indices: [usize; D]) -> Self {
        MultiKnotSpan { knots, indices }
    }

    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self, p: [usize; D]) -> RangeInclusive<usize> {
        // todo:
        // - get ranges per parametric direction
        // - build cartesian product
        // - flatten index
        
        todo!("")
    }
}

impl <T: RealField + Copy, const D: usize> MultiKnotVec<T, D> {

    /// Finds the [MultiKnotSpan] containing the parametric value `t`.
    ///
    /// # Arguments
    /// * `t` - Parametric values.
    /// * `n` - Number of basis functions in each parametric direction.
    ///
    pub fn find_span(&self, t: [T; D], n: [usize; D]) -> Result<MultiKnotSpan<T, D>, ()> {
        let indices = self.0.iter().enumerate()
            .flat_map(|(i, knots)| knots.find_span(t[i], n[i]))
            .map(|span| span.index)
            .collect_array()
            .ok_or(())?;
        
        Ok(MultiKnotSpan::new(self, indices))
    }
}