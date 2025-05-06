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
    
    /// The strides of the element indices. One less in each direction compared to the node indices.
    fn elem_strides(&self) -> Strides<usize, K> {
        Strides(self.strides.0.map(|nk| nk - 1))
    }
    
    pub fn elems(&self) {
        todo!("Return an iterator over elems (hyper rectangles)")
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
    pub fn from_dims(nx: usize, ny: usize) -> Self {
        TensorProd::new(Strides::from_dims([nx, ny]))
    }
}