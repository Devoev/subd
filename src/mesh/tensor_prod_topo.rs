//! Topology of a tensor product mesh.

use crate::cells::hyper_rectangle::HyperRectangle;
use crate::knots::index::{MultiIndex, Strides};
use itertools::Itertools;

pub struct TensorProd<const K: usize> {
    /// Dimensions in all `K` parametric directions.
    dims: [usize; K], // todo: replace dims with multi index range?
    /// Strides for each parametric direction.
    strides: Strides<usize, K>
}

impl<const K: usize> TensorProd<K> {
    /// Constructs a new [`TensorProd`] from the given `dims` and `strides`.
    pub fn new(dims: [usize; K], strides: Strides<usize, K>) -> Self {
        TensorProd { dims, strides }
    }

    /// Constructs a new [`TensorProd`] from the given `dims` and computes the strides itself.
    pub fn from_dims(dims: [usize; K]) -> Self {
        TensorProd::new(dims, Strides::from_dims(dims))
    }

    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> impl Iterator<Item = MultiIndex<usize, K>> {
        // todo: move this computation to Dims or MuliIndex Range or whatever
        let ranges = self.dims.map(|n| 0..n);
        ranges.into_iter()
            .multi_cartesian_product()
            .map(|idx| {
                let multi_idx = idx.into_iter().collect_array().unwrap();
                MultiIndex(multi_idx)
            })
    }

    /// Returns an iterator over all elements in lexicographical order.
    pub fn elems(&self) -> impl Iterator<Item=HyperRectangle<K>> {
        // todo: make this computation using the indices method
        let ranges = self.dims.map(|n| 0..n-1);
        ranges.into_iter()
            .multi_cartesian_product()
            .map(|idx| {
                let multi_idx = idx.into_iter().collect_array().unwrap();
                HyperRectangle(MultiIndex(multi_idx))
            })
    }
}