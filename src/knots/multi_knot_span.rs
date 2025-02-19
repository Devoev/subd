use crate::knots::knot_span::KnotSpan;
use crate::knots::multi_knot_vec::MultiKnotVec;
use itertools::{multipeek, Itertools, MultiProduct};
use nalgebra::RealField;
use std::iter::zip;
use std::ops::RangeInclusive;

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
    
    /// Creates [D] univariate [knot spans][KnotSpan].
    pub fn as_univariate(&self) -> [KnotSpan<T>; D] {
        zip(self.indices, &self.knots.0)
            .map(|(i, knots)| KnotSpan::new(knots, i))
            .collect_array()
            .unwrap()
    }

    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self, p: [usize; D]) -> MultiProduct<RangeInclusive<usize>> {
        self.as_univariate().iter().enumerate()
            .map(|(i, span)| span.nonzero_indices(p[i]))
            .multi_cartesian_product()
    }

    /// Returns an iterator over all linear indices of basis functions which are nonzero in this span.
    pub fn nonzero_lin_indices(&self, n: [usize; D], p: [usize; D]) -> impl Iterator<Item=usize> {
        self.nonzero_indices(p)
            .map(move |idx| {
                let multi_idx = idx.into_iter().collect_array().unwrap();
                MultiKnotVec::<T, D>::linear_index(multi_idx, n)
            })
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