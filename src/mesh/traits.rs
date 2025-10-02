use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Dyn, OMatrix, OPoint, Scalar};
use std::ops::Range;
use crate::cells::traits::ToElement;
use crate::element::traits::ElemAllocator;
use crate::mesh::ElemOfMesh;
use crate::mesh::vertex_storage::VertexStorage;

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

/// Mesh topology where every cell implements [`ToElement`].
pub trait CellElementTopology<T, Coords>: MeshTopology<Cell: ToElement<T, Coords::GeoDim, Coords = Coords>>
    where T: Scalar,
          Coords: VertexStorage<T>,
          DefaultAllocator: Allocator<Coords::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Coords, Self>> {}

impl <T, Coords, Cells> CellElementTopology<T, Coords> for Cells
    where T: Scalar,
          Coords: VertexStorage<T>,
          Cells: MeshTopology,
          CellOfMesh<Cells>: ToElement<T, Coords::GeoDim, Coords = Coords>,
          DefaultAllocator: Allocator<Coords::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Coords, Cells>>
{}
