use num_traits::{NumAssign, PrimInt};
use std::iter::{zip, Sum};

/// Strides of a [`D`]-variate [`MultiIndex`].
#[derive(Debug, Clone)]
pub struct Strides<I, const D: usize>([I; D]);

impl <I: PrimInt + NumAssign, const D: usize> Strides<I, D> {

    /// Calculates [`Strides`] of the given dimensions array `dims`.
    pub fn from_dims(mut dims: [I; D]) -> Self {
        dims.iter_mut().fold(I::one(), |acc, x| {
            let tmp = *x * acc;
            *x = acc;
            tmp
        });

        Strides(dims)
    }
}

impl<I, const D: usize> IntoIterator for Strides<I, D> {
    type Item = I;
    type IntoIter = std::array::IntoIter<I, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, I, const D: usize> IntoIterator for &'a Strides<I, D> {
    type Item = &'a I;
    type IntoIter = std::slice::Iter<'a, I>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// A [`D`]-variate [multi index](https://en.wikipedia.org/wiki/Multi-index_notation)
/// with index type [`I`].
#[derive(Debug, Clone)]
pub struct MultiIndex<I, const D: usize>(pub [I; D]);

impl <I: PrimInt + NumAssign + Sum, const D: usize> MultiIndex<I, D> {

    /// Converts this [`MultiIndex`] into a linear index,
    /// with the given `strides`.
    pub fn into_lin(self, strides: &Strides<I, D>) -> I {
        zip(self, strides).map(|(i, stride)| i * *stride).sum()
    }
}

impl <I, const D: usize> From<[I; D]> for MultiIndex<I, D> {

    fn from(value: [I; D]) -> Self {
        MultiIndex(value)
    }
}

impl<I, const D: usize> IntoIterator for MultiIndex<I, D> {
    type Item = I;
    type IntoIter = std::array::IntoIter<I, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, I, const D: usize> IntoIterator for &'a MultiIndex<I, D> {
    type Item = &'a I;
    type IntoIter = std::slice::Iter<'a, I>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}