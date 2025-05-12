use std::iter::zip;
use crate::index::dimensioned::DimShape;
use crate::mesh::tensor_prod_topo::TensorProd;
use itertools::{repeat_n, Itertools};
use nalgebra::RealField;

/// Cartesian mesh with [tensor product topology](TensorProd)
pub struct CartMesh<T: RealField, const K: usize> {
    /// Breakpoints for each parametric direction.
    pub breaks: [Vec<T>; K],

    /// Tensor product topology of the mesh.
    pub topology: TensorProd<K>
}

impl<T: RealField, const K: usize> CartMesh<T, K> {
    /// Constructs a new [`CartMesh`] from the given `breaks` and `topology`.
    ///
    /// # Panics
    /// If the shape of the `breaks` does not equal the shape of the `topology`,
    /// the function will panic.
    pub fn new(breaks: [Vec<T>; K], topology: TensorProd<K>) -> Self {
        let n_breaks = breaks.iter().map(|b| b.len()).collect_vec();
        let n_topo = topology.dim_shape.0;
        assert_eq!(n_breaks, n_topo,
                   "Shape of `breaks` (is {:?}) doesn't equal shape of `topology` (is {:?})",
                   n_breaks, n_topo);
        CartMesh { breaks, topology }
    }

    /// Constructs a new [`CartMesh`] from the given `breaks`.
    /// The tensor product topology is constructed from the shape of the breaks.
    pub fn from_breaks(breaks: [Vec<T>; K]) -> Self {
        let shape = breaks.iter().map(|b| b.len()).collect_array().unwrap();
        CartMesh { breaks, topology: TensorProd::from_dims(DimShape(shape)) }
    }

    // todo: change impl and signature of CartMesh::elems
    /// Returns an iterator over all elements in this mesh.
    pub fn elems(&self) -> impl Iterator<Item = Vec<[usize; K]>> {
        let offsets = repeat_n(0..=1, K)
            .multi_cartesian_product()
            .collect_vec();
        
        self.topology.elems()
            .map(move |e| {
                offsets.iter().map(move |offset| {
                    zip(offset, e.0).map(|(di, i)| i + di).collect_array::<K>().unwrap()
                }).collect_vec()
            })
    }
}