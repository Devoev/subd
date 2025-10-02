use crate::cells::traits::ToElement;
use crate::element::traits::ElemAllocator;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::ElemOfMesh;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Scalar};

/// Topology of a mesh consisting of cells.
///
/// A topological mesh is conceptually a collection of cells with some sort of connectivity relation.
/// This trait mainly provides iteration over all mesh cells using [`Self::into_cell_iter`].
/// The connectivity relation between cells is provided by each [`Self::Cell`].
pub trait CellTopology {
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
pub type CellOfMesh<Cells> = <Cells as CellTopology>::Cell;

/// Cell topology where every cell implements [`ToElement`].
pub trait ElementTopology<T, Coords>: CellTopology<Cell: ToElement<T, Coords::GeoDim, Coords = Coords>>
    where T: Scalar,
          Coords: VertexStorage<T>,
          DefaultAllocator: Allocator<Coords::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Coords, Self>> {}

impl <T, Coords, Cells> ElementTopology<T, Coords> for Cells
    where T: Scalar,
          Coords: VertexStorage<T>,
          Cells: CellTopology,
          CellOfMesh<Cells>: ToElement<T, Coords::GeoDim, Coords = Coords>,
          DefaultAllocator: Allocator<Coords::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Coords, Cells>>
{}
