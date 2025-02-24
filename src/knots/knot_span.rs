use crate::knots::index::MultiIndex;
use crate::bspline::spline_basis::SplineBasis;
use crate::bspline::basis::Basis;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use itertools::Itertools;
use nalgebra::RealField;
use std::iter::zip;
use std::ops::RangeInclusive;

/// A knot span `[xi[i], xi[i+1])`.
#[derive(Debug, Clone)]
pub struct KnotSpan<'a, Idx, S> {
    /// The B-spline function space.
    space: &'a S,
    
    /// The span index.
    pub index: Idx
}

impl<'a, Idx, K> KnotSpan<'a, Idx, K> {
    /// Constructs a new [`KnotSpan`].
    pub fn new(space: &'a K, index: Idx) -> Self {
        KnotSpan { space, index }
    }
}

/// A multivariate knot span.
pub type MultiKnotSpan<'a, T, const D: usize> = KnotSpan<'a, MultiIndex<usize, D>, MultiSplineBasis<T, D>>;

/// A univariate knot span.
pub type KnotSpan1<'a, T> = KnotSpan<'a, usize, SplineBasis<T>>;

/// A bivariate knot span.
pub type KnotSpan2<'a, T> = MultiKnotSpan<'a, T, 2>;

impl<'a, T: RealField + Copy> KnotSpan1<'a, T> {
    /// Finds the [`KnotSpan`] containing the given parametric value `t`.
    pub fn find(space: &'a SplineBasis<T>, t: T) -> Result<Self, ()> {
        let knots = &space.knots;
        if !knots.range().contains(&t) { return Err(()) }

        // todo: update the usage of space.n in this algorithm and DEBUG in general
        if t == knots[space.n + 1] {
            return Ok(KnotSpan::new(space, space.n - 1));
        }

        let idx = knots.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        let idx = match idx {
            Ok(i) => { i }
            Err(i) => { i - 1 }
        };
        Ok(KnotSpan::new(space, idx))
    }
    
    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self) -> RangeInclusive<usize> {
        self.index - self.space.p..=self.index
    }
}

impl<'a, T: RealField + Copy, const D: usize> MultiKnotSpan<'a, T, D> {
    /// Finds the [`KnotSpan`] containing the parametric value `t`.
    pub fn find(knots: &'a MultiSplineBasis<T, D>, t: [T; D]) -> Result<Self, ()> {
        let index = zip(&knots.0, t)
            .flat_map(|(knots, ti)| knots.find_span(ti))
            .map(|span| span.index)
            .collect_array()
            .ok_or(())?;

        Ok(KnotSpan::new(knots, MultiIndex(index)))
    }
    
    /// Returns an iterator of univariate [knot spans][KnotSpan] for each parametric direction.
    fn spans(&self) -> impl Iterator<Item=KnotSpan1<T>> {
        zip(&self.index, &self.space.0)
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
}