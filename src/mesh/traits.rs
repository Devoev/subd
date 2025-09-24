use std::marker::PhantomData;
use crate::cells::geo;
use crate::cells::geo::{Cell, CellAllocator};
use crate::cells::node::NodeIdx;
use crate::diffgeo::chart::ChartAllocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Point, RealField, Scalar};
use nalgebra::allocator::Allocator;

/// Topology of a mesh consisting of cells.
pub trait MeshTopology {
    /// Topological element in the mesh.
    type Elem; //: topo::Cell<Const<K>>; todo: add bound

    /// Element iterator.
    type ElemIter: Iterator<Item = Self::Elem>;

    /// Returns the total number of elements in this mesh.
    fn num_elems(&self) -> usize;

    /// Returns an iterator over all elements in this mesh.
    fn elem_iter(&self) -> Self::ElemIter;
}

/// Storage for the geometrical vertex points of a mesh.
pub trait VertexStorage<T: Scalar, M: DimName> where DefaultAllocator: Allocator<M> {
    /// Node iterator.
    type NodeIter: Iterator<Item = NodeIdx>;

    /// Returns the total number of nodes in this mesh.
    fn num_nodes(&self) -> usize;

    /// Iterates over all nodes in this mesh in ascending order.
    fn node_iter(&self) -> Self::NodeIter;

    /// Gets the vertex point of the `i`-th node in the mesh.
    fn vertex(&self, i: NodeIdx) -> OPoint<T, M>; // todo: possibly also allow for multi index?
}

impl <T: Scalar, M: DimName> VertexStorage<T, M> for Vec<OPoint<T, M>>
where DefaultAllocator: Allocator<M>
{
    type NodeIter = impl Iterator<Item = NodeIdx>;

    fn num_nodes(&self) -> usize {
        self.len()
    }

    fn node_iter(&self) -> Self::NodeIter {
        (0..self.len()).map(NodeIdx)
    }

    fn vertex(&self, i: NodeIdx) -> OPoint<T, M> {
        self[i.0]
    }
}

/// A mesh consisting of connected topological cells and vertex coordinates.
pub struct Mesh<T: Scalar, M: DimName, Coords: VertexStorage<T, M>, Cells: MeshTopology>
    where DefaultAllocator: Allocator<M>
{
    /// Coordinate storage.
    pub coords: Coords,

    /// Mesh cell topology.
    pub cells: Cells,

    _phantom_data: PhantomData<(T, M)>,
}

impl <T, M, Coords, Cells> Mesh<T, M, Coords, Cells>
where T: Scalar,
      M: DimName,
      Coords: VertexStorage<T, M>,
      Cells: MeshTopology,
      DefaultAllocator: Allocator<M>
{

}

// todo: update signature of elems and add more methods
//  maybe change lifetime parameter