//! Mesh data structures.

use std::marker::PhantomData;
use nalgebra::{DefaultAllocator, Scalar};
use nalgebra::allocator::Allocator;
use crate::cells::traits::{ElemOfCell, ToElement};
use crate::element::traits::ElemAllocator;
use crate::mesh::cell_topology::{ElementTopology, CellTopology};
use crate::mesh::vertex_storage::VertexStorage;

pub mod elem_vertex;
pub mod cartesian;
pub mod bezier;
pub mod cell_topology;
pub mod face_vertex;
pub mod incidence;
pub mod knot_mesh;
pub mod vertex_storage;

/// The most generic mesh type.
///
/// A mesh consists of connected topological cells [`Cells`], defining the partition of the domain
/// and vertex coordinates [`Verts`] defining the embedding into Euclidean space.
#[derive(Copy, Clone, Debug)]
pub struct Mesh<T, Verts, Cells> {
    /// Coordinate storage.
    pub coords: Verts,

    /// Mesh cell topology.
    pub cells: Cells,

    _phantom_data: PhantomData<T>,
}

impl <T, Verts, Cells> Mesh<T, Verts, Cells>
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: CellTopology,
      DefaultAllocator: Allocator<Verts::GeoDim>
{
    /// Constructs a new [`Mesh<T, Verts,Cells>`] with the given `coords` and `cells`.
    pub fn with_coords_and_cells(coords: Verts, cells: Cells) -> Self {
        Mesh { coords, cells, _phantom_data: PhantomData }
    }

    /// Returns the total number of elements or cells in `self`.
    pub fn num_elems(&self) -> usize {
        self.cells.len()
    }

    /// Returns the total number of nodes in `self`.
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    /// Consumes `self` and returns an iterator over all topological cells in this mesh.
    pub fn into_cell_iter(self) -> Cells::CellIter {
        self.cells.into_cell_iter()
    }
}

impl <'a, T, Verts, Cells> Mesh<T, Verts, Cells>
where T: Scalar,
      Verts: VertexStorage<T>,
      &'a Cells: 'a + CellTopology,
      DefaultAllocator: Allocator<Verts::GeoDim>
{
    /// Returns an iterator over all topological cells in this mesh.
    pub fn cell_iter(&'a self) -> <&'a Cells as CellTopology>::CellIter {
        self.cells.into_cell_iter()
    }
}

// todo: introduce Iterator types IntoElemIter, IntoElemCellIter...

/// The geometrical element of the [`Mesh<T,Verts,Cells>`].
pub type ElemOfMesh<T, Verts, Cells> = ElemOfCell<T, <Cells as CellTopology>::Cell, <Verts as VertexStorage<T>>::GeoDim>;

/// Allocator for a [`Mesh<T,Verts,Cells>`].
///
/// Combines the point allocator of the vertex storage and the [`ElemAllocator`] for the mesh elements.
pub trait MeshAllocator<T, Verts, Cells>: ElemAllocator<T, ElemOfMesh<T, Verts, Cells>>
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts> + ?Sized,
      DefaultAllocator: Allocator<Verts::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Verts, Cells>> {}

impl <T, Verts, Cells> MeshAllocator<T, Verts, Cells> for DefaultAllocator
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts> + ?Sized,
      DefaultAllocator: Allocator<Verts::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Verts, Cells>> {}

impl <T, Verts, Cells> Mesh<T, Verts, Cells>
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts>,
      DefaultAllocator: MeshAllocator<T, Verts, Cells>
{
    /// Consumes `self` and returns an iterator over all geometrical elements in this mesh.
    pub fn into_elem_iter(self) -> impl Iterator<Item = ElemOfMesh<T, Verts, Cells>> {
        self.cells.into_cell_iter().map(move |cell| cell.to_element(&self.coords))
    }

    /// Consumes `self` and returns an iterator over `(elem,cell)` pairs.
    pub fn into_elem_cell_iter(self) -> impl Iterator<Item = (ElemOfMesh<T, Verts, Cells>, Cells::Cell)>  {
        self.cells.into_cell_iter().map(move |cell| (cell.to_element(&self.coords), cell))
    }
}

impl <'a, T, Verts, Cells> Mesh<T, Verts, Cells>
where T: Scalar,
      Verts: VertexStorage<T>,
      &'a Cells: 'a + ElementTopology<T, Verts>,
      DefaultAllocator: MeshAllocator<T, Verts, &'a Cells>
{
    /// Returns an iterator over all geometrical elements in this mesh.
    pub fn elem_iter(&'a self) -> impl Iterator<Item = ElemOfMesh<T, Verts, &'a Cells>> {
        self.cell_iter().map(move |cell| cell.to_element(&self.coords))
    }

    /// Returns an iterator over `(elem,cell)` pairs.
    pub fn elem_cell_iter(&'a self) -> impl Iterator<Item = (ElemOfMesh<T, Verts, &'a Cells>, <&'a Cells as CellTopology>::Cell)>  {
        self.cell_iter().map(move |cell| (cell.to_element(&self.coords), cell))
    }
}