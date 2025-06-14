//! Topology of a tensor product mesh.

use crate::cells::hyper_rectangle::HyperRectangleTopo;
use crate::cells::node::NodeIdx;
use crate::index::dimensioned::{DimShape, Strides};
use crate::index::multi_index::MultiIndex;
use crate::mesh::topo;
use crate::mesh::topo::MeshTopology;

/// Topology of a [`K`]-dimensional tensor product (cartesian) mesh.
/// The topological structure is a regular grid, that can in 2D be schematically visualized as
/// ```text
///    ^
///    |
/// ny---   +---+---+---+
///    |    |   |   |   |
///    |    +---+---+---+
///    |    |   |   |   |
///    |    +---+---+---+
///    |    |   |   |   |
///  0---   +---+---+---+
///    |
///    +----|-----------|--->
///         0           nx
/// ```
/// with `nx` and `ny` being the shapes in `x` and `y` direction respectively.
pub struct TensorProd<const D: usize> {
    /// Shape of the parametric directions.
    pub dim_shape: DimShape<D>,
    /// Strides for each parametric direction.
    pub strides: Strides<D>
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
    
    /// Returns an iterator over all nodes with linear indices in increasing index order.
    pub fn nodes(&self) -> impl Iterator<Item =NodeIdx> + '_ {
        self.indices().map(|idx| NodeIdx(idx.into_lin(&self.strides)))
    }

    /// Returns an iterator over all elements in lexicographical order.
    pub fn elems(&self) -> impl Iterator<Item=HyperRectangleTopo<D>> {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.range().map(HyperRectangleTopo)
    }
}

impl<'a, const D: usize> MeshTopology<'a, D, HyperRectangleTopo<D>> for TensorProd<D> {
    type Nodes = impl Iterator<Item =NodeIdx>;
    type Elems = impl Iterator<Item = HyperRectangleTopo<D>>;

    fn num_nodes(&self) -> usize {
        self.dim_shape.len()
    }

    fn num_elems(&self) -> usize {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.len()
    }

    fn nodes(&'a self) -> Self::Nodes {
        self.nodes()
    }

    fn elems(&'a self) -> Self::Elems {
        self.elems()
    }
}