use num_traits::PrimInt;
use std::iter::{zip, Sum};

/// Strides of a [`D`]-variate [`MultiIndex`].
#[derive(Debug, Copy, Clone)]
pub struct Strides<I, const D: usize>(pub [I; D]);

impl <I: PrimInt, const D: usize> Strides<I, D> {

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
#[derive(Debug, Copy, Clone)]
pub struct MultiIndex<I, const D: usize>(pub [I; D]);

impl <I: PrimInt + Sum, const D: usize> IntoLinear<I, D> for MultiIndex<I, D> {

    fn into_lin(self, strides: Strides<I, D>) -> I {
        zip(self, strides).map(|(i, stride)| i * stride).sum()
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

/// Conversion into a linear index `I`.
pub trait IntoLinear<I, const D: usize> {

    /// Converts `self` into a linear index using the given `strides`.
    fn into_lin(self, strides: Strides<I, D>) -> I;
}

/// Blanked implementation for [`Self::linearize`].
pub trait Linearize<I, const D: usize>
where
    I: PrimInt + Sum,
    Self: Iterator + Sized,
    Self::Item: IntoLinear<I, D>
{
    /// Linearizes each multi index into a linear index of type [`I`].
    fn linearize(self, strides: Strides<I, D>) -> impl Iterator<Item=I> {
        self.map(move |idx| idx.into_lin(strides))
    }
}

impl<T, I, const D: usize> Linearize<I, D> for T
where
    T: Iterator,
    I: PrimInt + Sum,
    T::Item: IntoLinear<I, D>
{}