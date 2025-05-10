/// Types composed of [`D`] elements of type [`T`],
/// i.e. a type isomorphic to the fixed-sized array `[T; D]`.
pub trait Dimensioned<const D: usize, T> {
    /// Converts this type to an array of the underlying elements.
    fn into_arr(self) -> [T; D];
}

/// Shape of a [`D`]-variate array.
#[derive(Debug, Copy, Clone)]
pub struct DimShape<const D: usize>(pub [usize; D]);

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