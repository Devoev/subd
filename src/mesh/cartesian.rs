use crate::cells::hyper_rectangle::HyperRectangle;
use crate::index::dimensioned::DimShape;
use crate::mesh::tensor_prod_topo::TensorProd;
use itertools::Itertools;
use nalgebra::{Point, RealField};
use std::iter::zip;

/// Cartesian mesh with [tensor product topology](TensorProd).
/// The grid formed by the mesh nodes can in 2D be schematically visualized as
/// ```text
///        ^
///        |
/// by[ny]---   +---+---+---+
///        |    |   |   |   |
///        |    +---+---+---+
///        |    |   |   |   |
///        |    +---+---+---+
///        |    |   |   |   |
///  by[0]---   +---+---+---+
///        |
///        +----|-----------|--->
///           bx[0]       bx[nx]
/// ```
/// where `bx` and `by` are the breakpoints for the `x` and `y` direction respectively.
pub struct CartMesh<T: RealField, const K: usize> {
    /// Breakpoints for each parametric direction.
    pub breaks: [Vec<T>; K],

    /// Tensor product topology of the mesh.
    pub topology: TensorProd<K>
}

impl<T: RealField + Copy, const K: usize> CartMesh<T, K> {
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
    
    /// Constructs the vertex point at the given multi-index position `idx`.
    pub fn vertex(&self, idx: [usize; K]) -> Point<T, K> {
        zip(idx, &self.breaks)
            .map(|(i, breaks)| breaks[i])
            .collect_array::<K>()
            .unwrap()
            .into()
    }

    // todo: change signature of CartMesh::elems
    /// Returns an iterator over all elements in this mesh.
    pub fn elems(&self) -> impl Iterator<Item = HyperRectangle<T, K>> + '_ {
        self.topology.elems()
            .map(move |e| {
                let idx_a = e.0;
                let idx_b = idx_a.map(|i| i + 1);
                let a = self.vertex(idx_a);
                let b = self.vertex(idx_b);
                HyperRectangle { a: a.coords, b: b.coords }
            })
    }
}