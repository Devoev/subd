use crate::cells::traits::ToElement;
use crate::element::traits::{ElemAllocator, VolumeElement};
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{ElemOfMesh, MeshAllocator};
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
    
    /// Returns `true` if the mesh contains no cells.
    fn is_empty(&self) -> bool;

    /// Creates an iterator over all cells in this mesh.
    fn into_cell_iter(self) -> Self::CellIter;
}

/// The topological cell of the `Cells`.
pub type CellOfMesh<Cells> = <Cells as CellTopology>::Cell;

/// Cell topology where every cell implements [`ToElement`].
/// 
/// The cells are required to be compatible with the vertex storage `Verts`,
/// by having the same geometrical dimension [`Verts::GeoDim`]
/// and the same node index [`Verts::NodeIdx`].
pub trait ElementTopology<T, Verts>: CellTopology<Cell: ToElement<T, Verts::GeoDim, Node = Verts::NodeIdx>>
    where T: Scalar,
          Verts: VertexStorage<T>,
          DefaultAllocator: ElemAllocator<T, ElemOfMesh<T, Verts, Self>> {} // fixme: cant use MeshAllocator here, because it depends on ElementTopology => cyclical dependency

impl <T, Verts, Cells> ElementTopology<T, Verts> for Cells
    where T: Scalar,
          Verts: VertexStorage<T>,
          Cells: CellTopology,
          Cells::Cell: ToElement<T, Verts::GeoDim, Node = Verts::NodeIdx>,
          DefaultAllocator: ElemAllocator<T, ElemOfMesh<T, Verts, Cells>>
{}

/// Cell topology of volumetric elements.
///
/// Same as [`ElementTopology`] but with the extra requirement that every element is a [`VolumeElement`].
pub trait VolumetricElementTopology<T, Verts>: ElementTopology<T, Verts, Cell: ToElement<T, Verts::GeoDim, Elem: VolumeElement<T>>>
where T: Scalar,
      Verts: VertexStorage<T>,
      DefaultAllocator: MeshAllocator<T, Verts, Self> {}

impl <T, Verts, Cells> VolumetricElementTopology<T, Verts> for Cells
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts>,
      Cells::Cell: ToElement<T, Verts::GeoDim, Elem: VolumeElement<T>>,
      DefaultAllocator: MeshAllocator<T, Verts, Cells>
{}
