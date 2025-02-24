use crate::knots::index::MultiIndex;
use crate::bspline::spline_basis::SplineBasis;
use crate::bspline::basis::Basis;
use crate::bspline::multi_spline_basis::MultiSplineBasis;
use itertools::Itertools;
use nalgebra::RealField;
use std::iter::zip;
use std::ops::RangeInclusive;

/// A knot span represented by a knot index `i`.
///
/// For a knot vector `xi`,
/// the span is the half-open interval `[xi[i], xi[i+1])`
#[derive(Debug, Copy, Clone)]
pub struct KnotSpan<Idx>(pub(crate) Idx);

/// A multivariate knot span.
pub type MultiKnotSpan<const D: usize> = KnotSpan<MultiIndex<usize, D>>;

/// A univariate knot span.
pub type KnotSpan1 = KnotSpan<usize>;

/// A bivariate knot span.
pub type KnotSpan2 = MultiKnotSpan<2>;

impl KnotSpan1 {
    /// Finds the [`KnotSpan`] containing the given parametric value `t`.
    pub fn find<T: RealField + Copy>(space: &SplineBasis<T>, t: T) -> Result<Self, ()> {
        let knots = &space.knots;
        if !knots.range().contains(&t) { return Err(()) }

        // todo: update the usage of space.n in this algorithm and DEBUG in general
        if t == knots[space.n + 1] {
            return Ok(KnotSpan(space.n - 1));
        }

        let idx = knots.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        let idx = match idx {
            Ok(i) => { i }
            Err(i) => { i - 1 }
        };
        Ok(KnotSpan(idx))
    }
    
    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self, p: usize) -> RangeInclusive<usize> {
        self.0 - p..=self.0
    }
}

impl<const D: usize> MultiKnotSpan<D> {
    /// Finds the [`KnotSpan`] containing the parametric value `t`.
    pub fn find<T: RealField + Copy>(space: &MultiSplineBasis<T, D>, t: [T; D]) -> Result<Self, ()> {
        let index = zip(&space.0, t)
            .flat_map(|(space, ti)| space.find_span(ti))
            .map(|span| span.0)
            .collect_array()
            .ok_or(())?;

        Ok(KnotSpan(MultiIndex(index)))
    }

    /// Returns a range over all indices of basis functions
    /// which are nonzero in this span.
    pub fn nonzero_indices(&self, p: [usize; D]) -> impl Iterator<Item=MultiIndex<usize, D>> {
        zip(&self.0, p)
            .map(|(i, p)| KnotSpan(*i).nonzero_indices(p))
            .multi_cartesian_product()
            .map(|vec| MultiIndex(vec.into_iter().collect_array().unwrap()))
    }
}