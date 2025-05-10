use std::iter::zip;
use crate::index::dimensioned::{Dimensioned, Strides};

// todo: remove Dimensioned super-trait, because this prevents impls for e.g. (1, 1..5)

/// A [`D`]-variate [multi index](https://en.wikipedia.org/wiki/Multi-index_notation).
pub trait MultiIndex<const D: usize, I>: Dimensioned<D, I> {
    /// Converts `self` into a linear index using the given `strides`.
    fn into_lin(self, strides: Strides<D>) -> I;
}

impl Dimensioned<1, usize> for usize {
    fn into_arr(self) -> [usize; 1] {
        [self]
    }
}

impl MultiIndex<1, usize> for usize {
    fn into_lin(self, _: Strides<1>) -> usize {
        self
    }
}

impl<const D: usize> Dimensioned<D, usize> for [usize; D] {
    fn into_arr(self) -> [usize; D] {
        self
    }
}

impl<const D: usize> MultiIndex<D, usize> for [usize; D] {
    fn into_lin(self, strides: Strides<D>) -> usize {
        zip(self, strides.into_arr()).map(|(i, stride)| i * stride).sum()
    }
}

// todo: implement for Range arrays