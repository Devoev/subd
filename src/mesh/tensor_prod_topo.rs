//! Topology of a tensor product mesh.

use std::iter::Map;
use crate::cells::hyper_rectangle::HyperRectangleTopo;
use crate::cells::node::NodeIdx;
use crate::index::dimensioned::{DimShape, MultiRange, Strides};
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

impl <const D: usize> TensorProd<D> {
    /// Constructs a new [`TensorProd`] from the given `dim_shape` and `strides`.
    pub fn new(dim_shape: DimShape<D>, strides: Strides<D>) -> Self {
        TensorProd { dim_shape, strides }
    }

    /// Constructs a new [`TensorProd`] from the given `dim_shape` and computes the strides itself.
    pub fn from_dims(dim_shape: DimShape<D>) -> Self {
        TensorProd::new(dim_shape, Strides::from(dim_shape))
    }
}

/// An iterator over the linear nodes ([`NodeIdx`]) of a [`TensorProd<D>`] mesh.
pub struct NodesIter<'a, const D: usize> {
    iter: MultiRange<[usize; D]>,
    strides: &'a Strides<D>
}

impl <'a, const D: usize> NodesIter<'a, D> {
    /// Constructs a new [`NodesIter`] from the given `iter` and `strides`.
    pub fn new(iter: MultiRange<[usize; D]>, strides: &'a Strides<D>) -> Self {
        NodesIter { iter, strides }
    }

    /// Constructs a enw [`NodesIter`] from the given tensor-product `msh`.
    pub fn from_msh(msh: &'a TensorProd<D>) -> Self {
        NodesIter::new(msh.indices(), &msh.strides)
    }
}

impl<const D: usize> Iterator for NodesIter<'_, D> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|idx| NodeIdx(idx.into_lin(self.strides)))
    }
}

/// An iterator over the elements ([`HyperRectangleTopo<D>`]) of a [`TensorProd<D>`] mesh.
pub type ElemsIter<const D: usize> = Map<MultiRange<[usize; D]>, fn([usize; D]) -> HyperRectangleTopo<D>>;

impl<const D: usize> TensorProd<D> {
    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> MultiRange<[usize; D]> {
        self.dim_shape.multi_range()
    }

    /// Returns an iterator over all nodes with linear indices in increasing index order.
    pub fn nodes(&self) -> NodesIter<'_, D> {
        NodesIter::from_msh(self)
    }

    /// Returns an iterator over all elements in lexicographical order.
    pub fn elems(&self) -> ElemsIter<D> {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.multi_range().map(HyperRectangleTopo)
    }
}

impl<'a, const D: usize> MeshTopology<'a, D, HyperRectangleTopo<D>> for TensorProd<D> {
    type Nodes = NodesIter<'a, D>;
    type Elems = ElemsIter<D>;

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