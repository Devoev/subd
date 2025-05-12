//! Topology of a tensor product mesh.

use crate::cells::hyper_rectangle::HyperRectangle;
use itertools::Itertools;
use crate::index::dimensioned::{DimShape, Strides};

pub struct TensorProd<const K: usize> {
    /// Shape of the parametric directions.
    dim_shape: DimShape<K>,
    /// Strides for each parametric direction.
    strides: Strides<K>
}

impl<const K: usize> TensorProd<K> {
    /// Constructs a new [`TensorProd`] from the given `dims` and `strides`.
    pub fn new(dims: DimShape<K>, strides: Strides<K>) -> Self {
        TensorProd { dim_shape: dims, strides }
    }

    /// Constructs a new [`TensorProd`] from the given `dims` and computes the strides itself.
    pub fn from_dims(dims: DimShape<K>) -> Self {
        TensorProd::new(dims, Strides::from(dims))
    }

    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> impl Iterator<Item = [usize; K]> {
        // todo: move this computation to Dims or MuliIndex Range or whatever
        let ranges = self.dim_shape.0.map(|n| 0..n);
        ranges.into_iter()
            .multi_cartesian_product()
            .map(|idx| {
                idx.into_iter().collect_array().unwrap()
            })
    }

    /// Returns an iterator over all elements in lexicographical order.
    pub fn elems(&self) -> impl Iterator<Item=HyperRectangle<K>> {
        // todo: make this computation using the indices method
        let ranges = self.dim_shape.0.map(|n| 0..n-1);
        ranges.into_iter()
            .multi_cartesian_product()
            .map(|idx| {
                let multi_idx = idx.into_iter().collect_array().unwrap();
                HyperRectangle(multi_idx)
            })
    }
}