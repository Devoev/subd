use crate::cells::cartesian::{CartCell, CartCellIdx};
use crate::index::dimensioned::DimShape;
use crate::mesh::tensor_prod_topo::TensorProd;
use itertools::Itertools;
use nalgebra::{Point, RealField};
use std::iter::zip;
use crate::knots::breaks::Breaks;
use crate::mesh::tensor_prod_topo;
use crate::mesh::topo::MeshTopology;
use crate::mesh::traits::Mesh;

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
    pub breaks: [Breaks<T>; K],

    /// Tensor product topology of the mesh.
    pub topology: TensorProd<K>
}

impl<T: RealField + Copy, const K: usize> CartMesh<T, K> {
    /// Constructs a new [`CartMesh`] from the given `breaks` and `topology`.
    ///
    /// # Panics
    /// If the shape of the `breaks` does not equal the shape of the `topology`,
    /// the function will panic.
    pub fn new(breaks: [Breaks<T>; K], topology: TensorProd<K>) -> Self {
        let n_breaks = breaks.iter().map(|b| b.len()).collect_vec();
        let n_topo = topology.dim_shape.0;
        assert_eq!(n_breaks, n_topo,
                   "Shape of `breaks` (is {:?}) doesn't equal shape of `topology` (is {:?})",
                   n_breaks, n_topo);
        CartMesh { breaks, topology }
    }

    /// Constructs a new [`CartMesh`] from the given `breaks`.
    /// The tensor product topology is constructed from the shape of the breaks.
    pub fn from_breaks(breaks: [Breaks<T>; K]) -> Self {
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
}

impl<'a, T: RealField + Copy, const K: usize> Mesh<'a, T, [T; K], K, K> for CartMesh<T, K> {
    type Elem = CartCellIdx<K>;
    type GeoElem = CartCell<T, K>;
    type NodesIter = tensor_prod_topo::NodesIter<'a, K>;
    type ElemsIter = tensor_prod_topo::ElemsIter<K>;

    fn num_nodes(&self) -> usize {
        self.topology.num_nodes()
    }

    fn num_elems(&self) -> usize {
        self.topology.num_elems()
    }

    fn nodes(&'a self) -> Self::NodesIter {
        self.topology.nodes()
    }

    fn elems(&'a self) -> Self::ElemsIter {
        self.topology.elems()
    }

    fn geo_elem(&'a self, elem: Self::Elem) -> Self::GeoElem {
        CartCell::from_topo(elem, self)
    }
}