use std::iter::zip;
use crate::index::dimensioned::{Dimensioned, Strides};

// todo: remove Dimensioned super-trait, because this prevents impls for e.g. (1, 1..5)

/// A [`D`]-variate [multi index](https://en.wikipedia.org/wiki/Multi-index_notation).
pub trait MultiIndex<I, const D: usize>: Dimensioned<I, D> {
    /// Converts `self` into a linear index using the given `strides`.
    fn into_lin(self, strides: &Strides<D>) -> I;
}

impl MultiIndex<usize, 1> for usize {
    fn into_lin(self, _: &Strides<1>) -> usize {
        self
    }
}

impl MultiIndex<usize, 2> for (usize, usize) {
    fn into_lin(self, strides: &Strides<2>) -> usize {
        let strides = strides.0;;
        self.0 * strides[0] + self.1 * strides[1]
    }
}

impl MultiIndex<usize, 3> for (usize, usize, usize) {
    fn into_lin(self, strides: &Strides<3>) -> usize {
        let strides = strides.0;
        self.0 * strides[0] + self.1 * strides[1] + self.2 * strides[2]
    }
}

impl<const D: usize> MultiIndex<usize, D> for [usize; D] {
    fn into_lin(self, strides: &Strides<D>) -> usize {
        zip(self, strides.0).map(|(i, stride)| i * stride).sum()
    }
}

// todo: implement for Range arrays