use crate::knots::knot_span::UniKnotSpan;
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
    pub fn as_univariate(&self) -> [UniKnotSpan<T>; D] {
        zip(self.indices, &self.knots.0)
            .map(|(i, knots)| UniKnotSpan::new(knots, i))
            .collect_array()
            .unwrap()
    }

    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self) -> MultiProduct<RangeInclusive<usize>> {
        self.as_univariate().iter()
            .map(|span| span.nonzero_indices())
            .multi_cartesian_product()
    }

    /// Returns an iterator over all linear indices of basis functions which are nonzero in this span.
    pub fn nonzero_lin_indices(&self) -> impl Iterator<Item=usize> + '_ {
        self.nonzero_indices()
            .map(move |idx| {
                let multi_idx = idx.into_iter().collect_array().unwrap();
                MultiKnotVec::<T, D>::linear_index(multi_idx, self.knots.n())
            })
    }
}

impl <T: RealField + Copy, const D: usize> MultiKnotVec<T, D> {

    /// Finds the [MultiKnotSpan] containing the parametric value `t`.
    pub fn find_span(&self, t: [T; D]) -> Result<MultiKnotSpan<T, D>, ()> {
        let indices = zip(&self.0, t)
            .flat_map(|(knots, ti)| knots.find_span(ti))
            .map(|span| span.index)
            .collect_array()
            .ok_or(())?;
        
        Ok(MultiKnotSpan::new(self, indices))
    }
}