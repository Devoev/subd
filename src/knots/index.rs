use std::iter::zip;

/// Strides of a [`D`]-variate [`MultiIndex`].
#[derive(Debug, Copy, Clone)]
pub struct Strides<const D: usize>(pub [usize; D]);

impl <const D: usize> Strides<D> {
    /// Calculates [`Strides`] of the given dimensions array `dims`.
    pub fn from_dims(mut dims: [usize; D]) -> Self {
        dims.iter_mut().fold(1, |acc, x| {
            let tmp = *x * acc;
            *x = acc;
            tmp
        });

        Strides(dims)
    }
}

// todo: make this a trait similar to SliceIndex and MatrixIndex
/// A [`D`]-variate [multi index](https://en.wikipedia.org/wiki/Multi-index_notation).
#[derive(Debug, Copy, Clone)]
pub struct MultiIndex<const D: usize>(pub [usize; D]);

impl <const D: usize> MultiIndex<D> {
    /// Converts `self` into a linear index using the given `strides`.
    pub fn into_lin(self, strides: Strides<D>) -> usize {
        zip(self, strides.0).map(|(i, stride)| i * stride).sum()
    }
}

impl <const D: usize> From<[usize; D]> for MultiIndex<D> {

    fn from(value: [usize; D]) -> Self {
        MultiIndex(value)
    }
}

impl<const D: usize> IntoIterator for MultiIndex<D> {
    type Item = usize;
    type IntoIter = std::array::IntoIter<usize, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a MultiIndex<D> {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// todo: add new iterator for that instead of impl Iterator

/// Blanked implementation for [`Self::linearize`].
pub trait Linearize<const D: usize>: Iterator<Item=MultiIndex<D>> + Sized {
    /// Linearizes each multi index in the iterator into a linear index.
    fn linearize(self, strides: Strides<D>) -> impl Iterator<Item=usize> {
        self.map(move |idx| idx.into_lin(strides))
    }
}

impl<const D: usize, T: Iterator<Item=MultiIndex<D>>> Linearize<D> for T {}