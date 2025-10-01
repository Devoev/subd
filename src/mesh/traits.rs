use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Dyn, OMatrix, OPoint, Scalar};
use std::ops::Range;

/// Topology of a mesh consisting of cells.
///
/// A topological mesh is conceptually a collection of cells with some sort of connectivity relation.
/// This trait mainly provides iteration over all mesh cells using [`Self::into_cell_iter`].
/// The connectivity relation between cells is provided by each [`Self::Cell`].
pub trait MeshTopology {
    /// Topological cell in the mesh.
    type Cell; //: topo::Cell<Const<K>>; todo: add bound

    /// Cell iterator.
    type CellIter: Iterator<Item = Self::Cell>;

    /// Returns the total number of cells in `self`.
    fn len(&self) -> usize;

    /// Creates an iterator over all cells in this mesh.
    fn into_cell_iter(self) -> Self::CellIter;
}

/// The topological cell of the `Cells`.
pub type CellOfMesh<Cells> = <Cells as MeshTopology>::Cell;

/// Storage for the geometrical vertex points of a mesh.
///
/// Each vertex point of a mesh is represented by an [`OPoint<T,Self::GeoDim>`]
/// with [`Self::GeoDim`] being the dimension of the embedding Euclidean space.
///
/// Access to individual vertices is given by the [`Self::vertex`] method,
/// given an index of type [`Self::NodeIdx`]. This is usually either a `usize`
/// or a multi-index.
/// Iteration over all vertices can be achieved
/// by the [`Self::node_iter`] and [`Self::len`] methods.
pub trait VertexStorage<T: Scalar> where DefaultAllocator: Allocator<Self::GeoDim> {
    /// Dimension of the embedding Euclidean space.
    type GeoDim: DimName;

    /// Node index defining a global ordering of vertices.
    type NodeIdx;

    /// Node iterator.
    type NodeIter: Iterator<Item = Self::NodeIdx>;

    /// Returns the total number of nodes in `self`.
    fn len(&self) -> usize;

    /// Iterates over all nodes in this mesh in ascending order.
    fn node_iter(&self) -> Self::NodeIter;

    /// Gets the vertex point of the `i`-th node in the mesh.
    fn vertex(&self, i: Self::NodeIdx) -> OPoint<T, Self::GeoDim>; // todo: possibly also allow for multi index?
}

/// The node index of the vertex storage `Coords`.
pub type NodeIdx<T, Coords> = <Coords as VertexStorage<T>>::NodeIdx;

/// Vector of control points.
impl <T: Scalar, M: DimName> VertexStorage<T> for Vec<OPoint<T, M>>
where DefaultAllocator: Allocator<M>
{
    type GeoDim = M;
    type NodeIdx = usize;
    type NodeIter = Range<usize>;

    fn len(&self) -> usize {
        self.len()
    }

    fn node_iter(&self) -> Self::NodeIter {
        0..self.len()
    }

    fn vertex(&self, i: usize) -> OPoint<T, M> {
        self[i].clone()
    }
}

/// Matrix of row-wise control points.
impl <T: Scalar, M: DimName> VertexStorage<T> for OMatrix<T, Dyn, M>
where DefaultAllocator: Allocator<M>
{
    type GeoDim = M;
    type NodeIdx = usize;
    type NodeIter = Range<usize>;

    fn len(&self) -> usize {
        self.nrows()
    }

    fn node_iter(&self) -> Self::NodeIter {
        0..self.nrows()
    }

    fn vertex(&self, i: usize) -> OPoint<T, Self::GeoDim> {
        OPoint::from(self.row(i).transpose())
    }
}