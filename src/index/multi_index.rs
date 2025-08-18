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
        let strides = strides.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_lin() {
        //            shape = 4x4
        let strides = Strides([1, 4]);
        
        let idx = (2, 0);
        assert_eq!(idx.into_lin(&strides), 2);
        let idx = [2, 0];
        assert_eq!(idx.into_lin(&strides), 2);
        
        let idx = (1, 3);
        assert_eq!(idx.into_lin(&strides), 13);
        let idx = [1, 3];
        assert_eq!(idx.into_lin(&strides), 13);
        
        //            shape = 3x3x3
        let strides = Strides([1, 3, 9]);

        let idx = (0, 0, 0);
        assert_eq!(idx.into_lin(&strides), 0);
        let idx = [0, 0, 0];
        assert_eq!(idx.into_lin(&strides), 0);

        let idx = (1, 2, 0);
        assert_eq!(idx.into_lin(&strides), 7);
        let idx = [1, 2, 0];
        assert_eq!(idx.into_lin(&strides), 7);

        let idx = (1, 0, 2);
        assert_eq!(idx.into_lin(&strides), 19);
        let idx = [1, 0, 2];
        assert_eq!(idx.into_lin(&strides), 19);
    }
}