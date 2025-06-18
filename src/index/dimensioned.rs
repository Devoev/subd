use std::iter::Map;
use std::ops::Range;
use crate::index::multi_index::MultiIndex;
use itertools::{Itertools, MultiProduct};
use nalgebra::{Point, SVector, Scalar};

/// Types composed of [`D`] elements of type [`T`],
/// i.e. a type isomorphic to the fixed-sized array `[T; D]`.
pub trait Dimensioned<T, const D: usize> {
    /// Converts this type to an array of the underlying elements.
    fn into_arr(self) -> [T; D];
}

impl <T> Dimensioned<T, 1> for T {
    fn into_arr(self) -> [T; 1] {
        [self]
    }
}

impl <T> Dimensioned<T, 2> for (T, T) {
    fn into_arr(self) -> [T; 2] {
        [self.0, self.1]
    }
}

impl <T> Dimensioned<T, 3> for (T, T, T) {
    fn into_arr(self) -> [T; 3] {
        [self.0, self.1, self.2]
    }
}

impl<T, const D: usize> Dimensioned<T, D> for [T; D] {
    fn into_arr(self) -> [T; D] {
        self
    }
}

impl<const D: usize, T> Dimensioned<T, D> for SVector<T, D> {
    fn into_arr(self) -> [T; D] {
        let [inner] = self.data.0;
        inner
    }
}

impl<const D: usize, T: Scalar> Dimensioned<T, D> for Point<T, D> {
    fn into_arr(self) -> [T; D] {
        self.coords.into_arr()
    }
}

/// Shape of a [`D`]-variate array.
#[derive(Debug, Copy, Clone)]
pub struct DimShape<const D: usize>(pub [usize; D]);

/// An iterator over the multivariate cartesian product of ranges.
/// Yields all multi-indices inside a [`DimShape`].
pub type MultiRange<I> = Map<MultiProduct<Range<usize>>, fn(Vec<usize>) -> I>;

impl<const D: usize> DimShape<D> {
    /// Returns the total length of this shape, i.e. the product of all dimensions.
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    /// Returns an iterator over multi indices of type `I` in range of this shape.
    pub fn multi_range<I: MultiIndex<usize, D> + From<[usize; D]>>(&self) -> MultiRange<I> {
        let ranges = self.0.map(|n| 0..n);
        
        ranges.into_iter()
            .multi_cartesian_product()
            .map(|idx| {
                idx.into_iter().collect_array().unwrap().into()
            })
    }
    
    /// Reduces the size of every dimension in `self` by `num`.
    pub fn shrink(&mut self, num: usize) {
        self.0.iter_mut().for_each(|n| *n -= num);
    }
}

impl<const D: usize> Dimensioned<usize, D> for DimShape<D> {
    fn into_arr(self) -> [usize; D] {
        self.0
    }
}

/// Strides of a [`D`]-variate array.
#[derive(Debug, Copy, Clone)]
pub struct Strides<const D: usize>(pub [usize; D]);

impl<const D: usize> Dimensioned<usize, D> for Strides<D> {
    fn into_arr(self) -> [usize; D] {
        self.0
    }
}

impl <const D: usize> From<DimShape<D>> for Strides<D> {
    fn from(mut value: DimShape<D>) -> Self {
        value.0.iter_mut().fold(1, |acc, x| {
            let tmp = *x * acc;
            *x = acc;
            tmp
        });

        Strides(value.0)
    }
}