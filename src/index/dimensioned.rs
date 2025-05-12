use nalgebra::SVector;

/// Types composed of [`D`] elements of type [`T`],
/// i.e. a type isomorphic to the fixed-sized array `[T; D]`.
pub trait Dimensioned<const D: usize, T> {
    /// Converts this type to an array of the underlying elements.
    fn into_arr(self) -> [T; D];
}

impl<const D: usize, T: Clone> Dimensioned<D, T> for SVector<T, D> {
    fn into_arr(self) -> [T; D] {
        self.data.0[0].clone()
    }
}

/// Shape of a [`D`]-variate array.
#[derive(Debug, Copy, Clone)]
pub struct DimShape<const D: usize>(pub [usize; D]);

impl<const D: usize> DimShape<D> {
    /// Returns the total length of this shape, i.e. the product of all dimensions.
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }
}

impl<const D: usize> Dimensioned<D, usize> for DimShape<D> {
    fn into_arr(self) -> [usize; D] {
        self.0
    }
}

/// Strides of a [`D`]-variate array.
#[derive(Debug, Copy, Clone)]
pub struct Strides<const D: usize>([usize; D]);

impl<const D: usize> Dimensioned<D, usize> for Strides<D> {
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