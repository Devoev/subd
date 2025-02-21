use crate::knots::index::{MultiIndex, Strides};
use crate::knots::knot_vec::KnotVec;
use crate::knots::multi_knot_vec::MultiKnotVec;
use itertools::Itertools;
use nalgebra::RealField;
use std::iter::zip;
use std::ops::RangeInclusive;
use crate::knots::knots_trait::Knots;

/// A knot span `[xi[i], xi[i+1])`.
#[derive(Debug, Clone)]
pub struct KnotSpan<'a, Idx, K> {
    /// The knot vector.
    knots: &'a K,
    
    /// The span index.
    pub index: Idx
}

impl<'a, Idx, K> KnotSpan<'a, Idx, K> {
    /// Constructs a new [`KnotSpan`].
    pub fn new(knots: &'a K, index: Idx) -> Self {
        KnotSpan { knots, index }
    }
}

/// A multivariate knot span.
pub type MultiKnotSpan<'a, T, const D: usize> = KnotSpan<'a, MultiIndex<usize, D>, MultiKnotVec<T, D>>;

/// A univariate knot span.
pub type KnotSpan1<'a, T> = KnotSpan<'a, usize, KnotVec<T>>;

/// A bivariate knot span.
pub type KnotSpan2<'a, T> = KnotSpan<'a, [usize; 2], MultiKnotVec<T, 2>>;

impl<'a, T: RealField + Copy> KnotSpan1<'a, T> {
    /// Finds the [`KnotSpan`] containing the given parametric value `t`.
    pub fn find(knots: &'a KnotVec<T>, t: T) -> Result<Self, ()> {
        if t < knots.first() || t > knots.last() { return Err(()) }

        if t == knots[knots.n + 1] {
            return Ok(KnotSpan::new(knots, knots.n - 1));
        }

        let idx = knots.vec.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        let idx = match idx {
            Ok(i) => { i }
            Err(i) => { i - 1 }
        };
        Ok(KnotSpan::new(knots, idx))
    }
    
    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self) -> RangeInclusive<usize> {
        self.index - self.knots.p..=self.index
    }
}

impl<'a, T: RealField + Copy, const D: usize> MultiKnotSpan<'a, T, D> {
    /// Finds the [`KnotSpan`] containing the parametric value `t`.
    pub fn find(knots: &'a MultiKnotVec<T, D>, t: [T; D]) -> Result<Self, ()> {
        let index = zip(&knots.0, t)
            .flat_map(|(knots, ti)| knots.find_span(ti))
            .map(|span| span.index)
            .collect_array()
            .ok_or(())?;

        Ok(KnotSpan::new(knots, MultiIndex(index)))
    }
    
    /// Returns an iterator of univariate [knot spans][KnotSpan] for each parametric direction.
    fn spans(&self) -> impl Iterator<Item=KnotSpan1<T>> {
        zip(&self.index, &self.knots.0)
            .map(|(i, knots)| KnotSpan1::new(knots, *i))
    }

    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self) -> impl Iterator<Item=MultiIndex<usize, D>> {
        self.spans()
            .map(|span| span.nonzero_indices())
            .multi_cartesian_product()
            .map(|vec| MultiIndex(vec.into_iter().collect_array().unwrap()))
    }

    /// Returns an iterator over all linear indices of basis functions which are nonzero in this span.
    pub fn nonzero_lin_indices(&self) -> impl Iterator<Item=usize> + '_ {
        // todo: put a strides() function in multi knot vec or somewhere else
        let strides = Strides::from_dims(self.knots.n());
        self.nonzero_indices()
            .map(move |idx| idx.into_lin(&strides))
    }
}