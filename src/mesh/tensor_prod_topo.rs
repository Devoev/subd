//! Topology of a tensor product mesh.

use crate::cells::hyper_rectangle::HyperRectangle;
use crate::index::dimensioned::{DimShape, Strides};

pub struct TensorProd<const D: usize> {
    /// Shape of the parametric directions.
    dim_shape: DimShape<D>,
    /// Strides for each parametric direction.
    strides: Strides<D>
}

impl<const D: usize> TensorProd<D> {
    /// Constructs a new [`TensorProd`] from the given `dim_shape` and `strides`.
    pub fn new(dim_shape: DimShape<D>, strides: Strides<D>) -> Self {
        TensorProd { dim_shape, strides }
    }

    /// Constructs a new [`TensorProd`] from the given `dim_shape` and computes the strides itself.
    pub fn from_dims(dim_shape: DimShape<D>) -> Self {
        TensorProd::new(dim_shape, Strides::from(dim_shape))
    }

    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> impl Iterator<Item = [usize; D]> {
        self.dim_shape.range()
    }

    /// Returns an iterator over all elements in lexicographical order.
    pub fn elems(&self) -> impl Iterator<Item=HyperRectangle<D>> {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.range().map(|i: [usize; D]| HyperRectangle(i))
    }
}