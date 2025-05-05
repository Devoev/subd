//! Topology of a tensor product mesh.

use crate::knots::index::Strides;

pub struct TensorProd<const K: usize> {
    /// Strides for each parametric direction.
    strides: Strides<usize, K>
}

impl<const K: usize> TensorProd<K> {
    /// Constructs a new [`TensorProd`] from the given `strides`.
    pub fn new(strides: Strides<usize, K>) -> Self {
        TensorProd { strides }
    }
}

impl TensorProd<3> {
    /// Constructs a new [`TensorProd`] from the given dimensions `nx`, `ny` and `nz`.
    pub fn from_dims(nx: usize, ny: usize, nz: usize) -> Self {
        TensorProd::new(Strides::from_dims([nx, ny, nz]))
    }
}

impl TensorProd<2> {
    /// Constructs a new [`TensorProd`] from the given dimensions `nx` and `ny`.
    pub fn from_dims(nx: usize, ny: usize, nz: usize) -> Self {
        TensorProd::new(Strides::from_dims([nx, ny]))
    }
}